"""
AWS Integration module for Business Entity Resolution pipeline.
Provides helper functions for Amazon S3 dataset sync, artifact management, and credential checks.
"""
import os
import sys
from typing import List, Dict, Optional

from src.config import (
    AWS_REGION, S3_BUCKET, S3_KEYS,
    DATASET_DIR, OUTPUT_DIR, MODEL_DIR,
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    CANDIDATE_PAIRS_PATH, MATCHING_RESULTS_PATH,
    MODEL_PATH, MODEL_METADATA_PATH
)

HAS_BOTO3 = False
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
    HAS_BOTO3 = True
except ImportError:
    pass


def get_s3_client():
    """Initialize and return a boto3 S3 client."""
    if not HAS_BOTO3:
        raise ImportError(
            "[AWS Error] Python library 'boto3' is not installed. "
            "Please install it using: pip install boto3"
        )
    return boto3.client("s3", region_name=AWS_REGION)


def check_aws_credentials() -> bool:
    """Verify that valid AWS credentials are set up."""
    if not HAS_BOTO3:
        print("[AWS Error] 'boto3' module is missing. Please run 'pip install boto3'.")
        return False
        
    try:
        sts = boto3.client("sts", region_name=AWS_REGION)
        identity = sts.get_caller_identity()
        print(f"[AWS Credentials] Verified! Account: {identity.get('Account')}, Arn: {identity.get('Arn')}")
        return True
    except (BotoCoreError, ClientError) as e:
        print("\n========================================")
        print("     AWS CREDENTIAL ERROR DETECTED     ")
        print("========================================")
        print("Unable to locate valid AWS credentials.")
        print("Please configure your AWS credentials using one of the following methods:")
        print("  1. Run 'aws configure' in your terminal")
        print("  2. Set environment variables:")
        print("     - AWS_ACCESS_KEY_ID")
        print("     - AWS_SECRET_ACCESS_KEY")
        print("     - AWS_REGION (optional, default: us-east-1)")
        print(f"Details: {e}")
        print("========================================\n")
        return False


def check_s3_bucket_access(bucket_name: str = S3_BUCKET) -> bool:
    """Verify that the configured S3 bucket exists and is accessible."""
    if not bucket_name:
        print("\n========================================")
        print("     S3 BUCKET CONFIGURATION ERROR      ")
        print("========================================")
        print("S3_BUCKET environment variable is not set!")
        print("Please export or set the S3_BUCKET variable:")
        print("  On Linux/Mac: export S3_BUCKET='my-entity-resolution-bucket'")
        print("  On Windows PowerShell: $env:S3_BUCKET='my-entity-resolution-bucket'")
        print("========================================\n")
        return False
        
    s3 = get_s3_client()
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f"[AWS S3] Bucket '{bucket_name}' verified and accessible.")
        return True
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        print("\n========================================")
        print("     S3 BUCKET ACCESS ERROR DETECTED    ")
        print("========================================")
        if error_code == "404":
            print(f"S3 Bucket '{bucket_name}' does NOT exist.")
        elif error_code == "403":
            print(f"Access Denied for S3 Bucket '{bucket_name}'. Please check IAM permissions.")
        else:
            print(f"Error accessing bucket '{bucket_name}': {e}")
        print("========================================\n")
        return False


def download_file_from_s3(s3_key: str, local_path: str, bucket_name: str = S3_BUCKET) -> bool:
    """Download a file from S3 to local storage."""
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        print(f"[AWS S3 Download] s3://{bucket_name}/{s3_key} --> {local_path}")
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"[AWS S3 Download] Completed successfully! Saved to {local_path}")
        return True
    except ClientError as e:
        print(f"[AWS S3 Error] Failed to download s3://{bucket_name}/{s3_key}: {e}")
        return False


def upload_file_to_s3(local_path: str, s3_key: str, bucket_name: str = S3_BUCKET) -> bool:
    """Upload a local file to S3 storage."""
    if not os.path.exists(local_path):
        print(f"[AWS S3 Error] Cannot upload non-existent local file: {local_path}")
        return False
        
    s3 = get_s3_client()
    try:
        print(f"[AWS S3 Upload] {local_path} --> s3://{bucket_name}/{s3_key}")
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"[AWS S3 Upload] Successfully uploaded to s3://{bucket_name}/{s3_key}")
        return True
    except ClientError as e:
        print(f"[AWS S3 Error] Failed to upload {local_path} to s3://{bucket_name}/{s3_key}: {e}")
        return False


def download_dataset_from_s3(bucket_name: str = S3_BUCKET) -> bool:
    """Download all required train & test TSV files from S3 to local dataset/ directory."""
    if not check_aws_credentials() or not check_s3_bucket_access(bucket_name):
        return False
        
    print(f"\n[AWS S3] Downloading full dataset from s3://{bucket_name}/ ...")
    file_map = {
        S3_KEYS["train_source1"]: TRAIN_SOURCE1,
        S3_KEYS["train_source2"]: TRAIN_SOURCE2,
        S3_KEYS["train_source3"]: TRAIN_SOURCE3,
        S3_KEYS["train_ground_truth"]: TRAIN_GROUND_TRUTH,
        S3_KEYS["test_source1"]: TEST_SOURCE1,
        S3_KEYS["test_source2"]: TEST_SOURCE2,
        S3_KEYS["test_source3"]: TEST_SOURCE3,
    }
    
    success = True
    for s3_key, local_path in file_map.items():
        if not download_file_from_s3(s3_key, local_path, bucket_name):
            success = False
            
    return success


def upload_results_to_s3(bucket_name: str = S3_BUCKET) -> bool:
    """Upload matching results and candidate pairs to S3."""
    if not check_aws_credentials() or not check_s3_bucket_access(bucket_name):
        return False
        
    print(f"\n[AWS S3] Uploading execution outputs to s3://{bucket_name}/ ...")
    success = True
    if os.path.exists(CANDIDATE_PAIRS_PATH):
        if not upload_file_to_s3(CANDIDATE_PAIRS_PATH, S3_KEYS["candidate_pairs"], bucket_name):
            success = False
    if os.path.exists(MATCHING_RESULTS_PATH):
        if not upload_file_to_s3(MATCHING_RESULTS_PATH, S3_KEYS["matching_results"], bucket_name):
            success = False
    return success


def upload_model_to_s3(bucket_name: str = S3_BUCKET) -> bool:
    """Upload trained model binary and metadata to S3."""
    if not check_aws_credentials() or not check_s3_bucket_access(bucket_name):
        return False
        
    print(f"\n[AWS S3] Uploading trained model artifacts to s3://{bucket_name}/ ...")
    success = True
    if os.path.exists(MODEL_PATH):
        if not upload_file_to_s3(MODEL_PATH, S3_KEYS["model_binary"], bucket_name):
            success = False
    if os.path.exists(MODEL_METADATA_PATH):
        if not upload_file_to_s3(MODEL_METADATA_PATH, S3_KEYS["model_metadata"], bucket_name):
            success = False
    return success


def list_s3_objects(prefix: str = "", bucket_name: str = S3_BUCKET) -> List[Dict[str, str]]:
    """List S3 objects in bucket matching prefix."""
    s3 = get_s3_client()
    try:
        res = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        objects = []
        for obj in res.get("Contents", []):
            objects.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "last_modified": str(obj["LastModified"])
            })
        return objects
    except ClientError as e:
        print(f"[AWS S3 Error] Unable to list objects in s3://{bucket_name}/{prefix}: {e}")
        return []
