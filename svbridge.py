import os
import glob as glob_module
import argparse
import json
import asyncio
import logging
import logging.config
from datetime import datetime, timedelta, timezone
from threading import RLock
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI, Request, HTTPException, Depends, Header, APIRouter
from fastapi.responses import StreamingResponse
from apscheduler.schedulers.background import BackgroundScheduler

# Configurations
CONFIG_FILE = "svbridge-config.json"
DEFAULT_CONFIG: dict[str, str | bool | int | None] = {
    "port": 8086,
    "bind": "localhost",
    "key": "",
    "auto_refresh": True,
    "filter_model_names": True,
}

LOCATION = "global"
ENDPOINT_ID = "openapi"
PUBLISHERS = (
    "google",
    "anthropic",
    "meta",
)  # No api to list them all, you can manually add them
MODEL_NAMES_FILTER = (
    "google/gemini-",
    "anthropic/claude-",
    "meta/llama",
)  # Usually you wouldnt want to sift through hundreds of irrelevant ones

TOKEN_EXPIRY_BUFFER = timedelta(minutes=10)
BACKGROUND_INTERVAL = 5  # minutes


@dataclass
class Credential:
    """Represents a single Google Cloud credential"""
    file_path: str
    project_id: str | None = None
    access_token: str | None = None
    token_expiry: datetime | None = None

    def is_valid(self) -> bool:
        """Check if token is valid"""
        if not self.access_token or not self.token_expiry:
            return False
        now = datetime.now(timezone.utc)
        return now < (self.token_expiry - TOKEN_EXPIRY_BUFFER)

    def refresh(self) -> bool:
        """Refresh the token for this credential"""
        from google.auth import load_credentials_from_file
        from google.auth.transport.requests import Request

        try:
            credentials, project = load_credentials_from_file(
                self.file_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            credentials.refresh(Request())
            self.access_token = credentials.token
            self.token_expiry = credentials.expiry.replace(tzinfo=timezone.utc)
            if not self.project_id:
                self.project_id = project
            logger.info(f"[Credential] Refreshed token for {os.path.basename(self.file_path)} (project: {self.project_id})")
            return True
        except Exception as e:
            logger.error(f"[Credential] Failed to refresh {os.path.basename(self.file_path)}: {e}")
            return False

    def get_token(self) -> str | None:
        """Get valid token, refresh if needed"""
        if not self.is_valid():
            if not self.refresh():
                return None
        return self.access_token


@dataclass
class CredentialPool:
    """Manages multiple credentials with round-robin cycling"""
    credentials: list[Credential] = field(default_factory=list)
    current_index: int = 0
    lock: RLock = field(default_factory=RLock)

    def add(self, cred: Credential):
        """Add a credential to the pool"""
        self.credentials.append(cred)

    def count(self) -> int:
        """Number of credentials in pool"""
        return len(self.credentials)

    def get_current(self) -> Credential | None:
        """Get current credential"""
        if not self.credentials:
            return None
        with self.lock:
            return self.credentials[self.current_index]

    def rotate(self) -> Credential | None:
        """Rotate to next credential and return it"""
        if not self.credentials:
            return None
        with self.lock:
            self.current_index = (self.current_index + 1) % len(self.credentials)
            cred = self.credentials[self.current_index]
            logger.info(f"[Pool] Rotated to credential {self.current_index + 1}/{len(self.credentials)}: {os.path.basename(cred.file_path)}")
            return cred

    def get_token(self) -> tuple[str, str] | tuple[None, None]:
        """Get token from current credential, returns (token, project_id)"""
        cred = self.get_current()
        if not cred:
            return None, None
        token = cred.get_token()
        if token:
            return token, cred.project_id
        return None, None

    def refresh_all(self):
        """Refresh all credentials"""
        for cred in self.credentials:
            cred.refresh()


# Global state
credential_pool = CredentialPool()
config: dict[str, str | bool | int | None] = {}
token_lock = RLock()
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to load config on startup"""
    await startup_event()
    yield
    await shutdown_event()


app = FastAPI(lifespan=lifespan)
router = APIRouter()
logging.config.dictConfig(LOGGING_CONFIG)
logger: logging.Logger = logging.getLogger("uvicorn")


def load_config():
    """Load or initialize the config file"""
    global config
    with token_lock:
        is_changed = True
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    json_data = json.load(f)
                    config.update(json_data)
                    if config == json_data:
                        is_changed = False
                logger.info(f'[Config] Loaded <== "{CONFIG_FILE}"')
            except json.JSONDecodeError as e:
                logger.error(f"[Config] Failed to load config and using defaults: {e}")
                config = DEFAULT_CONFIG.copy()
        else:
            logger.warning(f"[Config] No config file found, using defaults")
            config = DEFAULT_CONFIG.copy()
        for k, v in DEFAULT_CONFIG.items():
            config.setdefault(k, v)

        if is_changed:
            save_config()


def save_config():
    """Save config file"""
    with token_lock:
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f'[Config] Saved ==> "{CONFIG_FILE}"')
        except Exception as e:
            logger.error(f"[Config] Failed to save config: {e}")


def discover_credentials() -> list[str]:
    """Discover credential JSON files in current directory"""
    # Look for service account JSON files (exclude config file)
    json_files = glob_module.glob("*.json")
    credential_files = []

    for f in json_files:
        if f == CONFIG_FILE:
            continue
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
                # Check if it looks like a service account file
                if data.get("type") == "service_account" and "private_key" in data:
                    credential_files.append(f)
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(credential_files)


def init_credential_pool():
    """Initialize the credential pool from discovered files or env var"""
    global credential_pool

    # Check for GOOGLE_APPLICATION_CREDENTIALS env var first
    env_cred = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_cred and os.path.exists(env_cred):
        cred = Credential(file_path=env_cred)
        if cred.refresh():
            credential_pool.add(cred)
            logger.info(f"[Pool] Added credential from env: {os.path.basename(env_cred)}")

    # Discover additional credentials in current directory
    discovered = discover_credentials()
    for cred_file in discovered:
        # Skip if already added from env
        abs_path = os.path.abspath(cred_file)
        if env_cred and os.path.abspath(env_cred) == abs_path:
            continue

        cred = Credential(file_path=abs_path)
        if cred.refresh():
            credential_pool.add(cred)
            logger.info(f"[Pool] Added credential: {cred_file} (project: {cred.project_id})")

    if credential_pool.count() == 0:
        logger.error("[Pool] No valid credentials found!")
    else:
        logger.info(f"[Pool] Initialized with {credential_pool.count()} credential(s)")


async def verify_token(authorization: str | None = Header(None)):
    """Verify the Bearer token if key is set"""
    auth_key = config.get("key")
    if auth_key:  # Only check if key is set
        if not authorization:
            logger.warning("[Auth] Missing Authorization header")
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning(
                f"[Auth] Invalid Authorization header format: {authorization}"
            )
            raise HTTPException(
                status_code=401, detail="Invalid Authorization header format"
            )

        token = parts[1]
        if token != auth_key:
            logger.warning("[Auth] Invalid token")
            raise HTTPException(status_code=401, detail="Invalid token")
        logger.info("[Auth] Token verified successfully")


@app.get("/")
async def root():
    return "Hello, this is Simple Vertex Bridge! UwU"


def get_endpoint_url(project_id: str) -> str:
    """Get the Vertex AI endpoint URL"""
    if LOCATION == "global":
        # Global region uses different URL format (no region prefix on domain)
        return (
            f"https://aiplatform.googleapis.com/v1"
            f"/projects/{project_id}"
            f"/locations/{LOCATION}"
            f"/endpoints/{ENDPOINT_ID}"
            f"/chat/completions"
        )
    else:
        # Regional endpoint
        return (
            f"https://{LOCATION}-aiplatform.googleapis.com/v1"
            f"/projects/{project_id}"
            f"/locations/{LOCATION}"
            f"/endpoints/{ENDPOINT_ID}"
            f"/chat/completions"
        )


@router.api_route(
    "/chat/completions",
    methods=["GET", "POST"],
    dependencies=[Depends(verify_token)],
)
async def chat_completions(request: Request):
    """Proxy to Vertex AI with Bearer token and round-robin failover"""
    logger.info(f"[Proxy] Received request: {request.url.path}")

    body = await request.body()

    # Try each credential in round-robin fashion
    num_credentials = credential_pool.count()
    if num_credentials == 0:
        raise HTTPException(status_code=500, detail="No credentials available")

    last_error = None
    for attempt in range(num_credentials):
        token, project_id = credential_pool.get_token()
        if not token or not project_id:
            logger.warning(f"[Proxy] Credential {attempt + 1} has no valid token, rotating...")
            credential_pool.rotate()
            continue

        target = get_endpoint_url(project_id)
        if request.url.query:
            target += f"?{request.url.query}"

        cred = credential_pool.get_current()
        cred_name = os.path.basename(cred.file_path) if cred else "unknown"
        logger.info(f"[Proxy] Attempt {attempt + 1}/{num_credentials} using {cred_name}")
        logger.info(f"[Proxy] {request.method} {target}")

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in ("host", "authorization", "content-length")
        }
        headers["Authorization"] = f"Bearer {token}"

        try:
            async with httpx.AsyncClient(http2=True, timeout=httpx.Timeout(300.0)) as client:
                async with client.stream(
                    request.method,
                    target,
                    headers=headers,
                    content=body,
                ) as resp:
                    status_code = resp.status_code
                    media_type = resp.headers.get("content-type") or "application/json"

                    # Check for rate limit error
                    if status_code == 429:
                        error_body = await resp.aread()
                        logger.warning(f"[Proxy] Got 429 from {cred_name}, rotating to next credential...")
                        credential_pool.rotate()
                        last_error = error_body
                        continue

                    # Success or other error - return response
                    async def stream_response():
                        async for chunk in resp.aiter_bytes():
                            yield chunk

                    # We need to collect the response since we're in a context manager
                    chunks = []
                    async for chunk in resp.aiter_bytes():
                        chunks.append(chunk)

                    async def yield_chunks():
                        for chunk in chunks:
                            yield chunk

                    return StreamingResponse(
                        yield_chunks(),
                        status_code=status_code,
                        media_type=media_type,
                    )

        except httpx.RequestError as e:
            logger.error(f"[Proxy] Request error with {cred_name}: {e}")
            credential_pool.rotate()
            last_error = str(e)
            continue

    # All credentials exhausted
    logger.error("[Proxy] All credentials exhausted")
    if last_error:
        return StreamingResponse(
            iter([last_error if isinstance(last_error, bytes) else last_error.encode()]),
            status_code=429,
            media_type="application/json",
        )
    raise HTTPException(status_code=500, detail="All credentials failed")


@router.api_route("/models", methods=["GET"], dependencies=[Depends(verify_token)])
async def models(request: Request):
    """Fetches available models from Vertex and returns them in OpenAI format"""
    logger.info(f"[Models] Received request: {request.url.path}")

    token, project_id = credential_pool.get_token()
    if not token or not project_id:
        logger.error("[Models] No valid token for models request")
        raise HTTPException(status_code=500, detail="Failed to obtain token")

    # Get all publishers asynchronously
    logger.info(f"[Models] Fetching models from {len(PUBLISHERS)} publishers...")

    async def retry_request(session, publisher, url, headers, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = await session.get(url, headers=headers)
                logger.info(f"[Models] {response.status_code} {publisher}")
                return response
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f'[Models] Failed to fetch models for publisher "{publisher}", will retry in 200ms: {type(e).__name__}'
                        + (f", {e}" if str(e) else "")
                    )
                    await asyncio.sleep(0.2)
                    continue
                return e

    assert http_client
    tasks = [
        retry_request(
            http_client,
            publisher,
            f"https://us-central1-aiplatform.googleapis.com/v1beta1/publishers/{publisher}/models",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
                "x-goog-user-project": project_id,
            },
        )
        for publisher in PUBLISHERS
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert to OpenAI format
    all_models = []
    for publisher, resp in zip(PUBLISHERS, responses):
        if isinstance(resp, Exception):
            logger.warning(
                f'[Models] Failed to fetch models for publisher "{publisher}": {type(resp).__name__}'
                + (f", {resp}" if str(resp) else "")
            )
            continue
        assert isinstance(resp, httpx.Response)
        if resp.status_code == 200:
            data = resp.json()
            publisher_models = data.get("publisherModels", [])
            for model in publisher_models:
                name = model.get("name")
                if name:
                    parts = name.split("/")
                    if (
                        len(parts) == 4
                        and parts[0] == "publishers"
                        and parts[2] == "models"
                    ):
                        model_publisher = parts[1]
                        model_name = parts[3]
                        model_id = f"{model_publisher}/{model_name}"
                        all_models.append(
                            {
                                "id": model_id,
                                "object": "model",
                                "owned_by": model_publisher,
                            }
                        )
        else:
            logger.warning(
                f'[Models] Failed to fetch models for publisher "{publisher}": {resp.status_code} {resp.text}.'
            )

    # Prefix filter
    if config.get("filter_model_names", True):
        all_models_count = len(all_models)
        all_models = [
            model
            for model in all_models
            if any(model["id"].startswith(prefix) for prefix in MODEL_NAMES_FILTER)
        ]
        logger.info(f"[Models] Fetched {len(all_models)}/{all_models_count} models")
    else:
        logger.info(f"[Models] Fetched {len(all_models)} models")

    return {"object": "list", "data": all_models}


app.include_router(router)
app.include_router(router, prefix="/v1")


async def startup_event():
    """Startup event"""
    global http_client
    logger.info("[HTTPClient] Creating reusable client...")
    http_client = httpx.AsyncClient(http2=True, timeout=None)
    logger.info("[HTTPClient] Created reusable client")

    # Initialize credential pool
    init_credential_pool()

    load_config()
    if config.get("auto_refresh"):
        logger.info(
            f"[Background] Started checking token every {BACKGROUND_INTERVAL} minutes"
        )
        scheduler = BackgroundScheduler()
        scheduler.add_job(credential_pool.refresh_all, "interval", minutes=BACKGROUND_INTERVAL)
        scheduler.start()


async def shutdown_event():
    """Shutdown event"""
    global http_client
    if http_client:
        await http_client.aclose()
        logger.info("[HTTPClient] Closed reusable client")
        http_client = None


def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="Simple Vertex Bridge /UwU")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        nargs="?",
        const=8086,
        help="Port to listen on (default: 8086)",
    )
    parser.add_argument(
        "-b",
        "--bind",
        type=str,
        nargs="?",
        const="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        nargs="?",
        const="",
        help="Specify the API key, if not set (as default), accept any key",
    )

    # Boolean flags for auto-refresh
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        "--auto-refresh",
        action=argparse.BooleanOptionalAction,
        dest="auto_refresh",
        help="Background token refresh check every 5 minutes (default: on)",
    )

    # Boolean flags for model filtering
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--filter-model-names",
        action=argparse.BooleanOptionalAction,
        dest="filter_model_names",
        help="Filtering common model names (default: on)",
    )

    args = parser.parse_args()

    load_config()
    config_updated = False
    for key, value in vars(args).items():
        if value is not None:  # Check if the argument was actually passed
            if config.get(key) != value:
                config[key] = value
                config_updated = True
    if config_updated:
        save_config()

    bind = config.get("bind")
    port = config.get("port")
    key = config.get("key")
    assert isinstance(bind, str)
    assert isinstance(port, int)
    assert isinstance(key, str)

    logger.info(f"--------")
    logger.info(f"Server: http://{bind}:{port}")
    if bind not in ("localhost", "127.0.0.1", "::1") and not key:
        logger.warning(f"[Auth] Server is exposed to the internet, PLEASE SET A KEY!")
    elif key:
        logger.info(f'API key: "{key}"')
    logger.info(f"--------")
    uvicorn.run("svbridge:app", host=bind, port=port)


if __name__ == "__main__":
    main()
