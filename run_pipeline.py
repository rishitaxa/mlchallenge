#!/usr/bin/env python
"""
Execution entry point for Business Entity Resolution Machine Learning Pipeline.
Supports both Local and AWS (S3 + EC2) execution modes.
"""
import os
import sys
import argparse

from src.config import (
    DATASET_DIR, OUTPUT_DIR, MODEL_DIR, RUN_MODE, S3_BUCKET,
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
)
from src.pipeline import run_pipeline
from src.aws_utils import (
    check_aws_credentials, check_s3_bucket_access,
    download_dataset_from_s3, upload_results_to_s3, upload_model_to_s3
)


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution ML Pipeline (Local & AWS S3/EC2 Execution)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "aws"],
        default=RUN_MODE,
        help="Execution mode: 'local' (read/write local filesystem) or 'aws' (read/write S3 bucket)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Pair batch size for feature extraction"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Validation set split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test dataset inference phase"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="AWS Mode: Force download of dataset from S3 before running pipeline"
    )
    parser.add_argument(
        "--upload-results",
        action="store_true",
        help="AWS Mode: Force upload of existing results to S3"
    )

    args = parser.parse_args()

    # Ensure required output & model directories exist locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("==================================================")
    print(f"  EXECUTION MODE SELECTED: {args.mode.upper()}")
    print("==================================================")

    if args.mode == "aws":
        print("[AWS Mode] Verifying AWS environment, credentials, and S3 bucket access...")
        if not check_aws_credentials() or not check_s3_bucket_access(S3_BUCKET):
            print("[AWS Error] AWS verification failed. Exiting pipeline.")
            sys.exit(1)

        # Handle standalone upload request if requested
        if args.upload_results:
            print("[AWS Mode] Uploading local output files and models to S3...")
            upload_results_to_s3(S3_BUCKET)
            upload_model_to_s3(S3_BUCKET)
            print("[AWS Mode] Upload completed successfully.")
            return

        # Check if local dataset files are missing or if force download requested
        required_dataset_files = [
            TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
            TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
        ]
        missing_dataset = any(not os.path.exists(p) for p in required_dataset_files)

        if missing_dataset or args.download_data:
            print("[AWS Mode] Downloading dataset files from Amazon S3...")
            success = download_dataset_from_s3(S3_BUCKET)
            if not success:
                print("[AWS Error] Failed to download complete dataset from S3!")
                sys.exit(1)

    else: # local mode
        # Validate local dataset directory
        train_dir = os.path.join(DATASET_DIR, "train")
        test_dir = os.path.join(DATASET_DIR, "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print("\n" + "=" * 60)
            print("ERROR: LOCAL DATASET NOT FOUND!")
            print("=" * 60)
            print(f"The required dataset directory '{DATASET_DIR}' was not found.")
            print("Please place your dataset files in the following structure:")
            print("""
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
""")
            print("Or run in AWS mode with environment variable set: S3_BUCKET='your-s3-bucket'")
            print("  python run_pipeline.py --mode aws --download-data")
            print("=" * 60 + "\n")
            sys.exit(1)

    # Execute full pipeline logic
    run_pipeline(
        val_size=args.val_size,
        random_seed=args.seed,
        batch_size=args.batch_size,
        skip_test_inference=args.skip_test
    )

    # If running in AWS mode, sync outputs back to S3 after pipeline completion
    if args.mode == "aws":
        print("\n[AWS Mode] Automatically uploading pipeline outputs & model to S3...")
        upload_results_to_s3(S3_BUCKET)
        upload_model_to_s3(S3_BUCKET)
        print("[AWS Mode] All outputs successfully synced to Amazon S3 bucket.")


if __name__ == "__main__":
    main()
