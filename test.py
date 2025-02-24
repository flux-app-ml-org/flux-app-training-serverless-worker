from loki_logger_handler.loki_logger_handler import LokiLoggerHandler
import logging
import os
import sys
import runpod
import boto3
    
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

LOKI_URL = os.getenv("LOKI_URL")

if LOKI_URL:
    logger.info("Configuring Loki logging.")
    loki_handler = LokiLoggerHandler(
        url=LOKI_URL,
        labels={"app": "flux-app-training-serverless-worker"}
    )
    logger.addHandler(loki_handler)
else:
    logger.warning("Loki credentials not provided, falling back to local logging.")
    
    local_handler = logging.StreamHandler(sys.stdout)
    local_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    local_handler.setFormatter(formatter)
    
    logger.addHandler(local_handler)

def upload_to_s3(file_path, bucket_name, s3_key):
    """
    Uploads a file to an S3 bucket.

    Args:
        file_path (str): The local path of the file to upload.
        bucket_name (str): The name of the S3 bucket.
        s3_key (str): The S3 key (path) to save the file under.

    Returns:
        str: The URL of the uploaded file.
    """
    s3 = boto3.resource('s3')
    # s3_client = boto3.client('s3')
    try:
        s3.Bucket().upload_file(file_path, s3_key)
        # s3_client.upload_file(file_path, 'directcut-flux-app', s3_key)
        logger.info("Uploaded file to S3", extra={"file_path": file_path, "s3_key": s3_key})
        return f"s3://{bucket_name}/{s3_key}"
    except Exception as e:
        logger.error("Failed to upload file to S3", extra={"file_path": file_path, "error": str(e)})
        return None

upload_to_s3('./test.txt', 'directcut-flux-app', 'test.txt')

runpod.