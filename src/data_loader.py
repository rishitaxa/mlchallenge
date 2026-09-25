"""
Data loading and statistics display module for Business Entity Resolution.
"""
import os
import json
import pandas as pd
from typing import Dict, Set, Tuple, List
from src.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
)


def validate_files_exist(file_paths: List[str]) -> None:
    """Ensure all required dataset files exist on disk."""
    missing = [path for path in file_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"The following required dataset files are missing:\n" +
            "\n".join(f"  - {p}" for p in missing) +
            "\nPlease ensure your dataset is located in the expected directory structure."
        )


def load_tsv(file_path: str) -> pd.DataFrame:
    """Load a TSV file reliably as string data type to preserve format."""
    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        on_bad_lines="skip"
    )
    # Strip column names whitespace
    df.columns = [col.strip() for col in df.columns]
    return df


def parse_matched_entity_ids(matched_str: str) -> Set[str]:
    """Parse matched_entity_ids string into a set of entity IDs cleanly."""
    if not matched_str or not matched_str.strip():
        return set()
    
    val = matched_str.strip()
    # Try parsing JSON array format if applicable
    if val.startswith("[") and val.endswith("]"):
        try:
            parsed = json.loads(val)
            return {str(item).strip() for item in parsed if str(item).strip()}
        except Exception:
            pass
            
    # Normalize separators (replace commas, semicolons, pipes with space)
    for sep in [",", ";", "|"]:
        val = val.replace(sep, " ")
        
    return {item.strip() for item in val.split() if item.strip()}


def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """
    Load train_ground_truth.tsv and return a dictionary mapping:
    source1_entity_id -> set(matched_entity_ids)
    """
    gt_df = load_tsv(file_path)
    
    # Identify column names
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    matched_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else gt_df.columns[1]
    
    gt_dict: Dict[str, Set[str]] = {}
    s1_vals = gt_df[s1_col].astype(str).values
    matched_vals = gt_df[matched_col].astype(str).values
    
    for s1_id, matched_str in zip(s1_vals, matched_vals):
        gt_dict[s1_id.strip()] = parse_matched_entity_ids(matched_str)
        
    return gt_dict


def print_summary_statistics(name: str, df: pd.DataFrame) -> None:
    """Print high-level summary statistics without exposing dataset contents."""
    print(f"\n--- Data Summary: {name} ---")
    print(f"Row count: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Check missing values
    missing_counts = {}
    for col in df.columns:
        # Count empty or whitespace-only strings as missing
        missing_n = (df[col].str.strip() == "").sum()
        missing_counts[col] = missing_n
    print(f"Missing values count per column: {missing_counts}")
    
    # Duplicate entity_id check
    if "entity_id" in df.columns:
        dups = df["entity_id"].duplicated().sum()
        print(f"Duplicate entity_ids count: {dups}")
        
    # Country distribution summary
    if "country" in df.columns:
        country_series = df["country"].str.strip().str.upper()
        unique_countries = country_series.nunique()
        print(f"Unique countries count: {unique_countries}")
        top_countries = country_series.value_counts().head(5).to_dict()
        print(f"Top 5 countries distribution: {top_countries}")
    print("-" * 35)


def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Set[str]]]:
    """Validate and load all training dataset files."""
    validate_files_exist([TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH])
    
    print("[DataLoader] Loading training sources and ground truth...")
    s1 = load_tsv(TRAIN_SOURCE1)
    s2 = load_tsv(TRAIN_SOURCE2)
    s3 = load_tsv(TRAIN_SOURCE3)
    gt = load_ground_truth(TRAIN_GROUND_TRUTH)
    
    print_summary_statistics("Train Source 1", s1)
    print_summary_statistics("Train Source 2", s2)
    print_summary_statistics("Train Source 3", s3)
    
    total_positives = sum(len(matches) for matches in gt.values())
    s1_with_matches = sum(1 for matches in gt.values() if len(matches) > 0)
    print(f"[DataLoader] Ground Truth Loaded: {len(gt):,} Source 1 entities registered.")
    print(f"[DataLoader] Total positive matching pairs in GT: {total_positives:,}")
    print(f"[DataLoader] Source 1 entities with at least 1 match: {s1_with_matches:,}")
    
    return s1, s2, s3, gt


def load_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate and load all test dataset files."""
    validate_files_exist([TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3])
    
    print("[DataLoader] Loading test sources...")
    s1 = load_tsv(TEST_SOURCE1)
    s2 = load_tsv(TEST_SOURCE2)
    s3 = load_tsv(TEST_SOURCE3)
    
    print_summary_statistics("Test Source 1", s1)
    print_summary_statistics("Test Source 2", s2)
    print_summary_statistics("Test Source 3", s3)
    
    return s1, s2, s3
