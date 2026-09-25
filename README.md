# Business Entity Resolution ML Pipeline (Local & AWS EC2 + S3)

A production-grade, high-performance Machine Learning pipeline designed for large-scale Business Entity Resolution (Record Linkage).

Given a deduplicated reference source (**Source 1**), the objective is to accurately identify all matching records from candidate sources (**Source 2** and **Source 3**) while optimizing for precision-heavy **F0.5** evaluation metric.

Supports both **Local execution** and **AWS Cloud Execution** (Amazon EC2 for compute + Amazon S3 for cloud object storage).

---

## ☁️ AWS System Architecture

```text
                                +-----------------------------------+
                                |          Amazon S3 Bucket         |
                                |       (Entity Resolution Data)    |
                                +-----------------------------------+
                                | dataset/train/*.tsv               |
                                | dataset/test/*.tsv                |
                                | output/candidate_pairs.tsv        |
                                | output/matching_results.tsv       |
                                | models/entity_resolution_model.pkl|
                                +-----------------+-----------------+
                                                  |
                                      Download    |    Upload
                                      Dataset     |    Outputs
                                                  v
                                +-----------------------------------+
                                |          Amazon EC2 Instance      |
                                |       (Compute & ML Pipeline)     |
                                +-----------------------------------+
                                | 1. Data Normalization             |
                                | 2. 5-Strategy Candidate Blocking  |
                                | 3. Batched Feature Engineering    |
                                | 4. LightGBM / XGBoost Model Train |
                                | 5. F0.5 Threshold Optimization    |
                                | 6. Test Inference Generation      |
                                +-----------------------------------+
```

### Amazon S3 Bucket Layout

```text
s3://<S3_BUCKET_NAME>/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── models/
│   ├── entity_resolution_model.pkl
│   └── model_metadata.json
└── output/
    ├── candidate_pairs.tsv
    └── matching_results.tsv
```

---

## 🛠️ Environment Setup & Installation

Ensure Python 3.8+ is installed. Install all required dependencies using:

```bash
pip install -r requirements.txt
```

---

## 💻 Local Execution Guide

1. Place your dataset in the local `dataset/` directory:

```text
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
```

2. Run the pipeline in local mode:

```bash
python run_pipeline.py --mode local
```

*Optional Local CLI flags:*
- `--batch-size 50000`: Batch size for memory-efficient feature engineering (default: `50000`)
- `--val-size 0.2`: Fraction of Source 1 entities allocated to validation split (default: `0.2`)
- `--seed 42`: Random seed for reproducibility (default: `42`)

---

## ☁️ AWS Cloud Execution Guide (EC2 + S3)

### 1. Required Environment Variables

Set your AWS environment variables before running in AWS mode:

**On Linux/macOS:**
```bash
export AWS_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
export S3_BUCKET="my-entity-resolution-bucket"
```

**On Windows PowerShell:**
```powershell
$env:AWS_REGION="us-east-1"
$env:AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY_ID"
$env:AWS_SECRET_ACCESS_KEY="YOUR_SECRET_ACCESS_KEY"
$env:S3_BUCKET="my-entity-resolution-bucket"
```

### 2. Required IAM Permissions

Your IAM user or EC2 Instance Role requires the following permissions for your S3 bucket:
- `s3:GetObject`
- `s3:PutObject`
- `s3:ListBucket`
- `s3:HeadBucket`

### 3. Step-by-Step AWS Setup & Workflow

#### Step A: Create S3 Bucket & Upload Dataset
Use the AWS CLI to create your S3 bucket and upload the local dataset:

```bash
aws s3 mb s3://my-entity-resolution-bucket --region us-east-1
aws s3 sync dataset/ s3://my-entity-resolution-bucket/dataset/
```

#### Step B: Launch & Configure Amazon EC2 Instance
1. Launch an EC2 instance (e.g. `c6i.2xlarge` or `m6i.2xlarge` running Ubuntu 22.04 LTS or Amazon Linux 2023).
2. Attach an IAM Role with S3 Access policies.
3. SSH into your EC2 instance:
   ```bash
   ssh -i key.pem ubuntu@ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com
   ```
4. Clone code repository and install dependencies:
   ```bash
   git clone <your-repo-url>
   cd ENTITY
   pip install -r requirements.txt
   ```

#### Step C: Run Pipeline in AWS Mode on EC2
To automatically pull the dataset from S3, run model training, and push final outputs back to S3:

```bash
export S3_BUCKET="my-entity-resolution-bucket"
python run_pipeline.py --mode aws --download-data
```

#### Step D: Retrieve Outputs from S3
Download generated predictions to your local machine:

```bash
aws s3 sync s3://my-entity-resolution-bucket/output/ output/
aws s3 sync s3://my-entity-resolution-bucket/models/ models/
```

---

## 🏗️ Architecture & Component Overview

```text
src/
├── __init__.py          # Package initializer
├── config.py            # Local & AWS configuration settings
├── aws_utils.py         # AWS S3 sync, bucket checking, and credential validation
├── data_loader.py       # Relative TSV loader, file validation, and summary stats
├── normalization.py     # Unicode, legal suffix, address, and token normalization
├── blocking.py          # 5-strategy candidate generation engine (UNION)
├── features.py          # Pairwise string, token, & sparse TF-IDF cosine feature engineering
├── labeling.py          # Ground truth labeling & hard negative sampling
├── model.py             # Supervised LightGBM/XGBoost model wrapper
├── evaluation.py        # F0.5 metric evaluation & grid-search threshold optimization
├── inference.py         # Test inference & output formatting
└── pipeline.py          # End-to-end pipeline execution coordinator
```

---

## 📊 Output Files

Output files are created under `output/` (local) and synced to `s3://<S3_BUCKET>/output/` (AWS):

1. `output/matching_results.tsv`:
   - `source1_entity_id`: Test Source 1 entity ID
   - `matched_entity_ids`: Space-separated matched candidate IDs (empty string if no match)

2. `output/candidate_pairs.tsv`:
   - `source1_entity_id`: Test Source 1 entity ID
   - `candidate_entity_ids`: Space-separated list of candidate IDs passed to the model
