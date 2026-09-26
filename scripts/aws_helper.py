#!/usr/bin/env python
import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import S3_BUCKET, S3_DATASET_PREFIX, S3_OUTPUT_PREFIX, S3_MODEL_PREFIX, DATASET_DIR, OUTPUT_DIR, MODEL_DIR
from src.aws_utils import (
    check_aws_credentials,
    check_s3_bucket_access,
    download_dataset_from_s3,
    upload_results_to_s3,
    upload_model_to_s3,
    list_s3_objects,
    get_s3_client
)
def upload_dataset_to_s3(bucket_name: str = S3_BUCKET) -> bool:
    if not check_aws_credentials() or not check_s3_bucket_access(bucket_name):
        return False
    print(f"\n[AWS Helper] Syncing local directory '{DATASET_DIR}/' to s3://{bucket_name}/{S3_DATASET_PREFIX}/ ...")
    s3 = get_s3_client()
    success = True
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.startswith("._") or file.endswith(".DS_Store"):
                continue
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, start=".").replace("\\", "/")
            s3_key = rel_path
            try:
                print(f"[AWS Helper Upload] {local_path} --> s3://{bucket_name}/{s3_key}")
                s3.upload_file(local_path, bucket_name, s3_key)
            except Exception as e:
                print(f"[AWS Helper Error] Failed to upload {local_path}: {e}")
                success = False
    if success:
        print(f"[AWS Helper] Successfully uploaded dataset to s3://{bucket_name}/{S3_DATASET_PREFIX}/")
    return success
def main():
    parser = argparse.ArgumentParser(description="AWS S3 Dataset & Artifact Management Helper Tool")
    parser.add_argument("--test-connection", action="store_true", help="Test AWS credentials and S3 bucket access")
    parser.add_argument("--upload-dataset", action="store_true", help="Upload local dataset/ directory to S3")
    parser.add_argument("--download-dataset", action="store_true", help="Download dataset from S3 to local dataset/")
    parser.add_argument("--upload-results", action="store_true", help="Upload local output/ and models/ to S3")
    parser.add_argument("--list-s3", action="store_true", help="List objects in S3 bucket")
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        return
    if args.test_connection:
        print("\n=== Testing AWS Connection ===")
        creds_ok = check_aws_credentials()
        bucket_ok = check_s3_bucket_access(S3_BUCKET) if creds_ok else False
        if creds_ok and bucket_ok:
            print("\n✅ AWS connection and S3 bucket access verified successfully!")
        else:
            print("\n❌ AWS connection test failed. Please verify credentials and S3_BUCKET environment variable.")
    if args.upload_dataset:
        upload_dataset_to_s3(S3_BUCKET)
    if args.download_dataset:
        download_dataset_from_s3(S3_BUCKET)
    if args.upload_results:
        upload_results_to_s3(S3_BUCKET)
        upload_model_to_s3(S3_BUCKET)
    if args.list_s3:
        print(f"\n=== Listing Objects in s3://{S3_BUCKET}/ ===")
        objects = list_s3_objects("", S3_BUCKET)
        if objects:
            for obj in objects:
                print(f"  - {obj['key']} ({obj['size']:,} bytes)")
        else:
            print("  (No objects found or bucket empty)")
if __name__ == "__main__":
    main()
