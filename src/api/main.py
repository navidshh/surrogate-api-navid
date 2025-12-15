
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi.openapi.models import OAuthFlows as OAuthFlowsModel
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Response, HTTPException, status, Depends, Form


from jose import jwt, jwk
from jose.utils import base64url_decode

from contextlib import asynccontextmanager
import os
# import asyncio
# import redis.asyncio as aioredis
import redis.asyncio as redis
import logging
import boto3
from .api_config import settings
from typing import Optional, Dict, Any, List
from .routes import auth, tests, maintenance, surrogate_model
from .auth.cognito import get_cognito_login_url
from .auth.dependency_functions import get_current_user, get_current_token, get_api_user
from .redis_client import init_redis, close_redis
import os

BASE_DIR = os.path.dirname(__file__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
redis_client = None  # this will hold the actual client

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Connecting to Redis...")
    try:
        app.state.redis_client = await init_redis()
        logger.info("Successfully connected to Redis!")
    except Exception as e:
        logger.warning(f"Redis not available, continuing without it: {e}")
        app.state.redis_client = None

    yield  # <-- app runs here

    # Shutdown
    if app.state.redis_client:
        logger.info("Closing Redis connection...")
        await close_redis(app.state.redis_client)

# app = FastAPI(lifespan=lifespan)
app = FastAPI(
    lifespan=lifespan,
    title="Surrogate Model API",
    swagger_ui_init_oauth={
        "clientId": settings.COGNITO_APP_PUBLIC_CLIENT_ID,
        "scopes": {"openid"},
        "usePkceWithAuthorizationCodeGrant": True,
    }
)


# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "static"))

# Register routers
app.include_router(auth.router, prefix="/auth", tags=["Auth"])
app.include_router(tests.router, prefix="/tests", tags=["Test"])
app.include_router(maintenance.router, prefix="/maintenance", tags=["Redis Maintenance"])
app.include_router(surrogate_model.router, prefix="/surrogate_model", tags=["Surrogate Model"])




# 1. Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(status_code=200, content={
        "status": "ok",
        "message": "Server is healthy"
    })

# 2. Main application endpoint
@app.get("/api/v1/data", response_class=JSONResponse)
async def get_sample_data(user: Optional[Dict[str, Any]] = Depends(get_current_user)):
# async def get_sample_data(user: Optional[Dict[str, Any]] = Depends(get_current_user)):get_api_user
    """Returns sample data to authenticated users."""
    # if not user:
    #     raise HTTPException(status_code=401, detail="Unauthorized")

    sample_data = {
        "message": f"Hello, {user.get('email', 'user')}!",
        "data": {
            "items": [
                {"id": 1, "value": "foo"},
                {"id": 2, "value": "bar"},
                {"id": 3, "value": "baz"},
            ]
        }
    }
    return JSONResponse(content=sample_data)

# --- Template routes ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    """
    Displays the home page.
    """
    context = {
    "request": request,
    "user": {},
    "cognito_login_url": get_cognito_login_url()
    }
    return templates.TemplateResponse("index.html", context)

# Root endpoint
@app.get("/entry")
async def root():
    return PlainTextResponse("Welcome to the application (v4.0.1). Try the /app or /health endpoints.")

@app.get("/profile", response_class=HTMLResponse, include_in_schema=False)
async def profile(request: Request, user: Optional[Dict[str, Any]] = Depends(get_current_user)):
    """
    Displays the profile page.
    """
    context = {
    "request": request,
    "user": user,
    "cognito_login_url": get_cognito_login_url()
    }
    return templates.TemplateResponse("profile.html", context)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)