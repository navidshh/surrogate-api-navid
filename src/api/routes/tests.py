# routes/test.py
import os
from fastapi import APIRouter, BackgroundTasks, Response, HTTPException, Header, Depends
from ..auth.dependency_functions import get_current_token, require_user, TokenInfo
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Depends, Request

import openpyxl
from io import BytesIO
from typing import Optional, Dict, Any, List
from ..auth.cognito import validate_token
import logging
import time
import redis.asyncio as redis
from redis import exceptions as redis_exceptions
from ..auth.dependency_functions import get_current_user
from ..api_config import settings
import sys
# from ..redis_client import redis_client


# logs of level INFO and higher to standard output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # If you want to log to a file as well, you can add a FileHandler here
        # logging.FileHandler("debug.log"),
        
        # This handler sends logs to the console
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

router = APIRouter()

def get_redis(request: Request):
    """Dependency that provides the Redis client."""
    return request.app.state.redis_client

@router.get("/test")
def test_endpoint():
    logger.info(f"Test endpoint")
    return {"message": "Test OK with logging"}

@router.get("/protected", response_model=dict)
async def protected_route(token_info: TokenInfo = Depends(get_current_token)):
    """
    Protected endpoint that supports both OAuth2 and Bearer token authentication.
    
    ## Authentication Methods:
    
    ### Method 1: Bearer Token
    - Click the "Authorize" button (lock icon)
    - Enter: `Bearer your_access_token_here`
    
    ### Method 2: OAuth2 Flow
    - Click the "Authorize" button
    - Use the OAuth2 Authorization Code flow
    
    ## Response:
    - Returns success message and token information
    """
    return {
        "message": "You have access!",
        "auth_method": token_info.auth_method,
        "token_type": token_info.token_type,
        "token_preview": f"{token_info.token[:20]}..." if len(token_info.token) > 20 else token_info.token
    }    


@router.post("/process-excel/")
async def process_excel(file: UploadFile = File(...)):
    # Read the uploaded file into memory
    contents = await file.read()
    input_stream = BytesIO(contents)

    # Load and process
    wb = openpyxl.load_workbook(input_stream)
    ws = wb.active
    ws["A1"] = "Processed by API"

    # Save to in-memory buffer
    output_stream = BytesIO()
    wb.save(output_stream)
    output_stream.seek(0)  # Reset cursor to start

    # Return file directly without saving
    return StreamingResponse(
        output_stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=processed.xlsx"}
    )    

    
@router.get("/logUserOut", summary="Logs the user out by clearing the session cookie.")
async def logout(response: Response):
    """
    Logs out the current user by clearing authentication tokens.
    Note: This clears local tokens. For complete Cognito logout, 
    redirect to the Cognito logout endpoint.
    """
    # Clear any local storage of tokens
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    response.delete_cookie("session_token")    

@router.get("/api/v1/user/groups", response_model=List[str], summary="Get user's Cognito groups")
async def get_user_groups(user: dict = Depends(require_user)):
    """
    Returns a list of Cognito groups for the currently authenticated user.
    """
    groups = user.get("cognito:groups", [])
    return groups

@router.get("/api/v1/user/groups2", response_model=List[str], summary="Get user's Cognito groups")
async def get_user_groups(token_info: TokenInfo = Depends(get_current_token)):
    """
    Returns a list of Cognito groups for the authenticated user.
    This endpoint accepts a Bearer token in the Authorization header.
    """
    # The get_current_token dependency gives us the raw token.
    # We now need to validate it to get the user claims.
    try:
        user_claims = validate_token(token_info.token)
    except HTTPException as e:
        # Re-raise the exception from validate_token if validation fails
        raise e

    # Once the token is validated, user_claims will contain the decoded token.
    # The rest of the logic is the same as before.
    groups = user_claims.get("cognito:groups", [])
    return groups       


# Redis connection check
@router.get("/redis_check")
async def redis_check(redis = Depends(get_redis)):
    try:
        reply = await redis.ping()
        if reply:  # True means PONG
            return JSONResponse(status_code=200, content={
                "status": "ok",
                "message": "Redis connection is healthy"
            })
        else:
            return JSONResponse(status_code=503, content={
                "status": "error",
                "message": "Redis ping failed",
                "details": str(reply)
            })
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "status": "error",
            "message": "Redis is unavailable",
            "details": str(e)
        })
    
@router.get("/ping")
async def ping_redis(redis = Depends(get_redis)):
    pong = await redis.ping()
    return {"status": "ok", "pong": pong}    
    

# --- Layer 1: Request Rate Limit Setup (MANUAL) ---
REQUEST_LIMIT_CONFIG = {
    # limit, window_in_seconds
    "Free-Tier": (1, 300),        # 1 request per 10 minutes (600 seconds)
    "Researcher-Tier": (1, 120),     # 
    "Developer-Tier": (1000, 86400),  # 1000 requests per day
}

# --- Layer 2: Concurrency Limit Setup (The Gatekeeper) ---
CONCURRENCY_QUOTAS = {
    "Free-Tier": 1,
    "Researcher-Tier": 1,
    "Developer-Tier": 1,
}

# --- Task Wrapper for Safe Decrement ---
async def task_wrapper(user_id: str, role: str, redis_client):
    concurrency_key = f"active_tasks:{user_id}"
    try:
        logger.info(f"Starting long running task for user '{user_id}' with role '{role}'...")
        time.sleep(5) 
        logger.info(f"Long running task for user '{user_id}' finished.")
    finally:
        if redis_client:
            logger.info(f"Decrementing concurrency counter for user {user_id}.")
            await redis_client.decr(concurrency_key)

async def is_redis_available(redis) -> bool:
    try:
        return await redis.ping()
    except redis_exceptions.RedisError:
        return False



@router.post("/rate-limited-task-simple/")
async def run_task_simple(
    background_tasks: BackgroundTasks,
    x_user_id: str = Header(...),
    x_user_role: str = Header("Free-Tier"),
    redis_client = Depends(get_redis)  # inject redis
):
    if not await is_redis_available(redis_client):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is temporarily unavailable (Redis down)."}
        )

    if x_user_role not in REQUEST_LIMIT_CONFIG:
        return JSONResponse(status_code=400, content={"detail": "Invalid user role specified."})
        
    # --- Check Layer 1: Manual Request Limit (The Shield) ---
    try:
        limit, window = REQUEST_LIMIT_CONFIG[x_user_role]
        
        # Create a key that is unique for the user AND the current time window
        current_timestamp = int(time.time())
        window_start_timestamp = int(current_timestamp / window) * window
        request_limit_key = f"requests:{x_user_role}:{x_user_id}:{window_start_timestamp}"
        
        # Use a pipeline to make the INCR and EXPIRE atomic
        pipe = redis_client.pipeline()
        pipe.incr(request_limit_key)
        pipe.expire(request_limit_key, time=window)
        results = await pipe.execute()
        
        current_hits = results[0]

        if current_hits > limit:
            logger.warning(f"Request limit shield hit for user {x_user_id}. Hits: {current_hits}, Limit: {limit}.")
            # Calculate remaining time in window for a more accurate Retry-After
            retry_after = window - (current_timestamp - window_start_timestamp)
            return JSONResponse(status_code=429, content={"detail": "API request rate limit exceeded."}, headers={"Retry-After": str(retry_after)})
            
        logger.info(f"Request limit check passed for {x_user_id}.")

    except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError):
        return JSONResponse(status_code=503, content={"detail": "Rate limiting service unavailable. Please try again later."})

    # --- Check Layer 2: Concurrency Limit (The Gatekeeper) ---
    concurrency_key = f"active_tasks:{x_user_id}"
    max_concurrency = CONCURRENCY_QUOTAS[x_user_role]
    
    current_count = await redis_client.incr(concurrency_key)
    
    try:
        if current_count > max_concurrency:
            logger.warning(f"Concurrency limit gatekeeper hit for user {x_user_id}. Has {current_count} active tasks, limit is {max_concurrency}.")
            await redis_client.decr(concurrency_key) # Revert the increment
            return JSONResponse(status_code=429, content={"detail": "You already have the maximum number of active tasks running."})
        
        logger.info(f"Concurrency check passed for user {x_user_id}. Starting task {current_count}/{max_concurrency}.")
        # background_tasks.add_task(task_wrapper, user_id=x_user_id, role=x_user_role, redis_client=redis_client)
        await task_wrapper(user_id=x_user_id, role=x_user_role, redis_client=redis_client)
        return {"status": "Task accepted and Finished."}
        
    except Exception as e:
        logger.error(f"An unexpected error occurred after incrementing concurrency counter: {e}. Reverting.")
        await redis_client.decr(concurrency_key)
        raise


@router.post("/rate-limited-task/")
async def run_task(
    background_tasks: BackgroundTasks,
    claims: dict = Depends(get_current_user),
    redis_client = Depends(get_redis)  # inject redis
):
    
    x_user_id = claims["sub"]    
    x_user_role = claims.get("cognito:groups", ["free-tier"])[0].lower()

    # If Redis is down entirely, we cannot do anything.
    if not await is_redis_available(redis_client):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is temporarily unavailable (Redis down)."}
        )

    if x_user_role not in REQUEST_LIMIT_CONFIG:
        return JSONResponse(status_code=400, content={"detail": "Invalid user role specified."})
        
    # --- Check Layer 1: Manual Request Limit (The Shield) ---
    try:
        limit, window = REQUEST_LIMIT_CONFIG[x_user_role]
        
        # Create a key that is unique for the user AND the current time window
        current_timestamp = int(time.time())
        window_start_timestamp = int(current_timestamp / window) * window
        request_limit_key = f"requests:{x_user_role}:{x_user_id}:{window_start_timestamp}"
        
        # Use a pipeline to make the INCR and EXPIRE atomic
        pipe = redis_client.pipeline()
        pipe.incr(request_limit_key)
        pipe.expire(request_limit_key, time=window)
        results = await pipe.execute()
        
        current_hits = results[0]

        if current_hits > limit:
            logger.warning(f"Request limit shield hit for user {x_user_id}. Hits: {current_hits}, Limit: {limit}.")
            # Calculate remaining time in window for a more accurate Retry-After
            retry_after = window - (current_timestamp - window_start_timestamp)
            return JSONResponse(status_code=429, content={"detail": "API request rate limit exceeded."}, headers={"Retry-After": str(retry_after)})
            
        logger.info(f"Request limit check passed for {x_user_id}.")

    except (redis_exceptions.ConnectionError, redis_exceptions.TimeoutError):
        return JSONResponse(status_code=503, content={"detail": "Rate limiting service unavailable. Please try again later."})

    # --- Check Layer 2: Concurrency Limit (The Gatekeeper) ---
    concurrency_key = f"active_tasks:{x_user_id}"
    max_concurrency = CONCURRENCY_QUOTAS[x_user_role]
    
    current_count = await redis_client.incr(concurrency_key)
    
    try:
        if current_count > max_concurrency:
            logger.warning(f"Concurrency limit gatekeeper hit for user {x_user_id}. Has {current_count} active tasks, limit is {max_concurrency}.")
            await redis_client.decr(concurrency_key) # Revert the increment
            return JSONResponse(status_code=429, content={"detail": "You already have the maximum number of active tasks running."})
        
        logger.info(f"Concurrency check passed for user {x_user_id}. Starting task {current_count}/{max_concurrency}.")
        # background_tasks.add_task(task_wrapper, user_id=x_user_id, role=x_user_role, redis_client=redis_client)
        await task_wrapper(user_id=x_user_id, role=x_user_role, redis_client=redis_client)
        return {"status": "Task accepted and Finished."}
        
    except Exception as e:
        logger.error(f"An unexpected error occurred after incrementing concurrency counter: {e}. Reverting.")
        await redis_client.decr(concurrency_key)
        raise    