from fastapi import APIRouter, Form, HTTPException, Depends, Request
from fastapi import BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from io import BytesIO
from datetime import datetime
import time

import json
import pandas as pd
from typing import List, Dict, Optional
from ..api_config import settings
from ..rate_limit_config import REQUEST_LIMIT_CONFIG, CONCURRENCY_QUOTAS

from run_model import run_predictions  # will not trigger Typer CLI
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from ..auth.dependency_functions import get_current_user
from redis import exceptions as redis_exceptions
from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError
import logging
import sys


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

s3 = boto3.client("s3")
BUCKET_NAME = settings.BUCKET_NAME

def get_redis(request: Request):
    """Dependency that provides the Redis client."""
    return request.app.state.redis_client

async def is_redis_available(redis) -> bool:
    try:
        return await redis.ping()
    except redis_exceptions.RedisError:
        return False

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    email: Optional[str] = Form(None, description="Optional user email address")
):
    try:

        # Validate file extension
        if not file.filename.lower().endswith('.xlsx'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only .xlsx files are permitted for upload. Please ensure your file is in the correct format and try again."
            )
                
        if email:
            # Sanitize email: lowercase, replace non-alphanumeric with '_', remove leading/trailing '_'
            sanitized_email = ''.join(c if c.isalnum() else '_' for c in email.lower()).strip('_')
            prefix = sanitized_email if sanitized_email else "general"  # Fallback if sanitization empties it
        else:
            prefix = "general"

        # Generate object key (overwrites if exists)
        # object_key = f"uploads/{prefix}_{file.filename}"
        object_key = f"uploads/{prefix}_input.xlsx"

        # Read file content into memory
        content = await file.read()

        # Upload to S3 (put_object overwrites existing keys by default)
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=object_key,
            Body=content
        )

        return {
            "status": "success",
            "file_name": file.filename,
            "s3_key": object_key
        }

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"S3 upload error: {str(e)}")
    

@router.post("/run-model-s3")
async def run_model_endpoint_s3(
    config_file: str = Form("input_config.yml", description="Configuration file name for the ML model (default: input_config.yml)"),
    email: Optional[str] = Form(None, description="Optional user email address")
):
    try:
        if email:
            # Sanitize email: lowercase, replace non-alphanumeric with '_', remove leading/trailing '_'
            sanitized_email = ''.join(c if c.isalnum() else '_' for c in email.lower()).strip('_')
            prefix = sanitized_email if sanitized_email else "general"  # Fallback if sanitization empties it
        else:
            prefix = "general"

        # Generate input object key
        input_filename = f"{prefix}_input.xlsx"
        input_key = f"uploads/{input_filename}"

        # Download from S3
        try:
            s3_response = s3.get_object(Bucket=BUCKET_NAME, Key=input_key)
            content = s3_response['Body'].read()
            input_stream = BytesIO(content)
            df = pd.read_excel(input_stream)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(
                    status_code=404,
                    detail=f"Input file not found in S3: {input_filename}. Please upload the file first and try again."
                )
            else:
                raise HTTPException(status_code=500, detail=f"S3 download error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading input file from S3: {str(e)}")

        # Prepare dfs dict (compatible with original: key is filename)
        dfs = {input_filename: df}

        # Call the ML logic - config_file should be the YAML config path, not the Excel filename
        results = run_predictions(config_file=config_file, api_mode=True, building_data_dict=dfs)

        if not results or "error" in results:
            raise HTTPException(status_code=500, detail=results.get("error", "ML prediction failed"))

        # Parse results
        json_results = {k: json.loads(v) for k, v in results.items()}

        # Add execution time
        json_results["execution_time"] = datetime.utcnow().isoformat() + "Z"  # UTC ISO format

        # Generate output object key
        output_filename = f"{prefix}_output.json"
        output_key = f"uploads/{output_filename}"

        # Upload to S3
        s3.put_object(
            Bucket=BUCKET_NAME,
            Key=output_key,
            Body=json.dumps(json_results),
            ContentType="application/json"
        )

        return {
            "status": "success",
            "input_file": input_filename,
            "output_key": output_key
        }

    except (BotoCoreError, ClientError) as e:
        raise HTTPException(status_code=500, detail=f"S3 operation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")    



@router.post("/run-model-s3-rate-limited")
async def run_task(
    claims: dict = Depends(get_current_user),
    redis_client = Depends(get_redis),
    config_file: str = Form(..., description="Configuration file name for the ML model")
):
    
    x_user_id = claims["sub"]    
    x_user_role = claims.get("cognito:groups", ["Free-Tier"])[0]

    if not await is_redis_available(redis_client):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service is temporarily unavailable (Redis down)."}
        )

    if x_user_role not in REQUEST_LIMIT_CONFIG:
        return JSONResponse(status_code=400, content={"detail": "Invalid user role specified."})
        
    # --- Check Layer 1: Manual Request Limit (The Shield) ---
    try:
        request_limit_config = REQUEST_LIMIT_CONFIG[x_user_role]
        limit = request_limit_config.requests
        window = request_limit_config.window_seconds        
        
        current_timestamp = int(time.time())
        window_start_timestamp = int(current_timestamp / window) * window
        request_limit_key = f"requests:{x_user_role}:{x_user_id}:{window_start_timestamp}"
        
        pipe = redis_client.pipeline()
        pipe.incr(request_limit_key)
        pipe.expire(request_limit_key, time=window)
        results = await pipe.execute()
        current_hits = results[0]

        if current_hits > limit:
            logger.warning(f"Request limit shield hit for user {x_user_id}. Hits: {current_hits}, Limit: {limit}.")
            retry_after = window - (current_timestamp - window_start_timestamp)
            return JSONResponse(
                status_code=429, 
                content={"detail": "API request rate limit exceeded."}, 
                headers={"Retry-After": str(retry_after)}
            )
            
        logger.info(f"Request limit check passed for {x_user_id}.")

    except (RedisConnectionError, RedisTimeoutError):
        return JSONResponse(status_code=503, content={"detail": "Rate limiting service unavailable. Please try again later."})

    # --- Check Layer 2: Concurrency Limit (The Gatekeeper) ---
    concurrency_key = f"active_tasks:{x_user_id}"
    max_concurrency_config = CONCURRENCY_QUOTAS[x_user_role]
    max_concurrency_limit = max_concurrency_config.max_concurrent
    
    current_count = await redis_client.incr(concurrency_key)
    
    # *** FIX: Use a try/finally block to guarantee the counter is decremented ***
    try:
        if current_count > max_concurrency_limit:
            # IMPROVEMENT: Log the numeric limit for clarity
            logger.warning(f"Concurrency limit hit for user {x_user_id}. Active tasks: {current_count}, Limit: {max_concurrency_limit}.")
            # The decrement is now in the finally block, but we must do it here too before exiting early.
            await redis_client.decr(concurrency_key) 
            return JSONResponse(status_code=429, content={"detail": "You already have the maximum number of active tasks running."})
        
        # IMPROVEMENT: Log the numeric limit for clarity
        logger.info(f"Concurrency check passed for user {x_user_id}. Starting task {current_count}/{max_concurrency_limit}.")        
        
        await process_user_model(
            user_email=claims.get("email", None),
            user_id=x_user_id,
            role=x_user_role,
            redis_client=redis_client,
            config_file=config_file
        )        
        return {"status": "Task accepted and Finished."}
        
    finally:
        # This will ALWAYS run, whether the task succeeds or fails, releasing the slot.
        # But we should not decrement if the check failed in the first place.
        if current_count <= max_concurrency_limit:
            logger.info(f"Task finished for {x_user_id}. Decrementing concurrency counter.")
            await redis_client.decr(concurrency_key)     
    

@router.post("/download-result")
async def download_result(
    email: Optional[str] = Form(None, description="Optional user email address")
):
    """
    This endpoint accepts an optional email in the request body (form format),
    sanitizes it (or uses "general" if not provided),
    reads the corresponding result file from S3, and returns the predictions in JSON format.
    """

    if email is None:
        sanitized = "general"
    else:
        # Sanitize email: lowercase, replace non-alphanumeric characters with '_' for filename safety
        sanitized_email = ''.join(c if c.isalnum() else '_' for c in email.lower()).strip('_')
        sanitized = sanitized_email if sanitized_email else "general"  # Fallback if sanitization empties it

    # Define the S3 key for the result JSON
    output_key = f"uploads/{sanitized}_output.json"

    # Download from S3
    try:
        s3_response = s3.get_object(Bucket=BUCKET_NAME, Key=output_key)
        content = s3_response['Body'].read().decode('utf-8')
        json_results = json.loads(content)
        return JSONResponse(status_code=200, content=json_results)
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            raise HTTPException(
                status_code=404,
                detail=f"Result file not found in S3 for key: {output_key}. Please run the model first (and/or wait) and try again."
            )
        else:
            raise HTTPException(status_code=500, detail=f"S3 download error: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing result file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


async def process_user_model(
    user_email: str,
    user_id: str,
    role: str,
    redis_client,
    config_file: str
):
    """
    Runs the actual model for the user, loads the Excel from S3,
    performs run_predictions(), and uploads result to S3.
    Always decrements concurrency counter at the end.
    """

    concurrency_key = f"active_tasks:{user_id}"

    try:
        # ---------------------------------------------------
        # 0. Sanitize prefix (same logic as your run_model_endpoint_s3)
        # ---------------------------------------------------
        if user_email:
            sanitized = ''.join(c if c.isalnum() else '_' for c in user_email.lower()).strip('_')
            if not sanitized:
                sanitized = "general"
        else:
            sanitized = "general"

        # S3 keys for input/output
        input_filename = f"{sanitized}_input.xlsx"
        input_key = f"uploads/{input_filename}"

        output_filename = f"{sanitized}_output.json"
        output_key = f"uploads/{output_filename}"

        # ---------------------------------------------------
        # 1. Download input file from S3
        # ---------------------------------------------------
        try:
            s3_response = s3.get_object(Bucket=BUCKET_NAME, Key=input_key)
            content = s3_response["Body"].read()
            input_stream = BytesIO(content)
            df = pd.read_excel(input_stream)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.error(f"Input file not found for user {user_id}: {input_filename}")
            else:
                logger.error(f"S3 download error for user {user_id}: {str(e)}")
            return  # Task ends but concurrency decrements inside finally

        # Build dfs dict (like your original endpoint)
        dfs = {input_filename: df}

        # ---------------------------------------------------
        # 2. Run model predictions
        # ---------------------------------------------------
        try:
            results = run_predictions(
                config_file=config_file,
                api_mode=True,
                building_data_dict=dfs
            )
        except Exception as e:
            logger.error(f"Model prediction error for {user_id}: {e}")
            return

        if not results or "error" in results:
            logger.error(f"Model returned error for {user_id}: {results.get('error')}")
            return

        # Parse JSON strings into objects
        json_results = {k: json.loads(v) for k, v in results.items()}
        json_results["execution_time"] = datetime.utcnow().isoformat() + "Z"

        # ---------------------------------------------------
        # 3. Upload output
        # ---------------------------------------------------
        try:
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=output_key,
                Body=json.dumps(json_results),
                ContentType="application/json"
            )
            logger.info(f"Model output uploaded for user {user_id}: {output_key}")
        except Exception as e:
            logger.error(f"Failed to upload output for user {user_id}: {e}")

    except Exception as e:
        logger.error(f"Unexpected error in process_user_model for {user_id}: {e}")

    finally:
        # ---------------------------------------------------
        # 4. ALWAYS decrement concurrency counter
        # ---------------------------------------------------
        try:
            await redis_client.decr(concurrency_key)
        except Exception:
            logger.error(f"Failed to decrement concurrency counter for {user_id}")



@router.post("/run-model", include_in_schema=False)
async def run_model_endpoint(
    config_file: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    This endpoint accepts a configuration file name and multiple Excel files,
    runs the ML model, and returns the predictions imediately in JSON format.
    Currently returns without processing (currently disabled)
    """
    return JSONResponse(status_code=200, content="ok-temporarily disabled")
    # The following code is commented out to avoid unreachable code warnings
    # while the endpoint is temporarily disabled. Uncomment when re-enabling.    

    # # ... (file reading logic is the same) ...
    # dfs: Dict[str, pd.DataFrame] = {}

    # for file in files:
    #     contents = await file.read()
    #     input_stream = BytesIO(contents)
    #     try:
    #         dfs[file.filename] = pd.read_excel(input_stream)
    #     except Exception as e:
    #         return JSONResponse(status_code=400, content={"error": f"Error reading {file.filename}: {e}"})

    # # Call the ML logic. 'results' is a dictionary of JSON strings.
    # results = run_predictions(config_file=config_file, api_mode=True, building_data_dict=dfs)

    # if not results or "error" in results:
    #     return JSONResponse(status_code=500, content=results)
        
    # # The results are valid JSON strings, so we parse them back into
    # # Python objects for a clean API response.
    # json_results = {k: json.loads(v) for k, v in results.items()}

    # # Pass the parsed Python objects to the response.
    # return JSONResponse(status_code=200, content=json_results)
