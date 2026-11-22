"""
Bhasha-Setu — S3 Storage Service
Handles uploading local files to AWS S3.
"""

import os
import re
import logging
import boto3
from botocore.exceptions import ClientError

from pipeline.config import AWS_REGION, S3_BUCKET

log = logging.getLogger(__name__)


def upload_to_s3(local_path: str, s3_key: str = None) -> str:
    """
    Upload a local file to the Bhasha-Setu S3 bucket.

    Args:
        local_path: Path to the local file to upload.
        s3_key:     S3 object key. Defaults to the file's sanitized basename.

    Returns:
        The S3 URI: s3://<bucket>/<key>
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")

    if s3_key is None:
        clean_base = os.path.basename(local_path.replace("\\", "/"))
        s3_key = re.sub(r"[^a-zA-Z0-9._-]", "_", clean_base)
    else:
        # Prevent traversal or unsafe S3 key prefixes
        s3_key = s3_key.replace("..", "").lstrip("/\\")

    client = boto3.client("s3", region_name=AWS_REGION)

    try:
        log.info(f"Uploading {local_path} → s3://{S3_BUCKET}/{s3_key}")
        client.upload_file(local_path, S3_BUCKET, s3_key)
        uri = f"s3://{S3_BUCKET}/{s3_key}"
        log.info(f"Upload complete: {uri}")
        return uri

    except ClientError as e:
        log.error(f"S3 upload ClientError: {e}")
        raise RuntimeError(f"S3 upload failed: {e}") from e
    except Exception as e:
        error_name = type(e).__name__
        if "Credential" in error_name or "NoCredentials" in error_name:
            raise RuntimeError(
                "AWS credentials not configured. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in environment "
                "or .streamlit/secrets.toml"
            ) from e
        elif "Endpoint" in error_name or "Connection" in error_name:
            raise RuntimeError(
                f"Cannot connect to AWS S3 in region '{AWS_REGION}'. "
                "Check your internet connection and AWS_DEFAULT_REGION setting."
            ) from e
        raise RuntimeError(f"S3 upload failed ({error_name}): {e}") from e


def delete_from_s3(s3_key: str) -> bool:
    """Delete an object from S3. Returns True on success."""
    try:
        client = boto3.client("s3", region_name=AWS_REGION)
        client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        log.info(f"Deleted s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        log.warning(f"Failed to delete S3 object {s3_key}: {e}")
        return False
