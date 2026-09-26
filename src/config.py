import os
RUN_MODE = os.environ.get("RUN_MODE", "local").lower()
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
TEST_DIR = os.path.join(DATASET_DIR, "test")
TEST_SOURCE1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3 = os.path.join(TEST_DIR, "test_source3.tsv")
OUTPUT_DIR = "output"
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "entity_resolution_model.pkl")
MODEL_METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_DATASET_PREFIX = os.environ.get("S3_DATASET_PREFIX", "dataset").strip("/")
S3_OUTPUT_PREFIX = os.environ.get("S3_OUTPUT_PREFIX", "output").strip("/")
S3_MODEL_PREFIX = os.environ.get("S3_MODEL_PREFIX", "models").strip("/")
S3_KEYS = {
    "train_source1": f"{S3_DATASET_PREFIX}/train/train_source1.tsv",
    "train_source2": f"{S3_DATASET_PREFIX}/train/train_source2.tsv",
    "train_source3": f"{S3_DATASET_PREFIX}/train/train_source3.tsv",
    "train_ground_truth": f"{S3_DATASET_PREFIX}/train/train_ground_truth.tsv",
    "test_source1": f"{S3_DATASET_PREFIX}/test/test_source1.tsv",
    "test_source2": f"{S3_DATASET_PREFIX}/test/test_source2.tsv",
    "test_source3": f"{S3_DATASET_PREFIX}/test/test_source3.tsv",
    "candidate_pairs": f"{S3_OUTPUT_PREFIX}/candidate_pairs.tsv",
    "matching_results": f"{S3_OUTPUT_PREFIX}/matching_results.tsv",
    "model_binary": f"{S3_MODEL_PREFIX}/entity_resolution_model.pkl",
    "model_metadata": f"{S3_MODEL_PREFIX}/model_metadata.json"
}
RANDOM_SEED = 42
PAIR_BATCH_SIZE = 50000
NAME_PREFIX_LEN = 4
MIN_TOKEN_LEN = 3
MAX_TOKEN_FREQ_RATIO = 0.03
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MIN_DF = 2
TFIDF_TOP_K = 15
NEGATIVE_RATIO = 5
HARD_NEGATIVE_RATIO = 0.7
VAL_SIZE = 0.2
BETA = 0.5
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
