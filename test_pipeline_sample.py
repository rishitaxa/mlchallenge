import os
import sys
import numpy as np
import pandas as pd
from src.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
from src.data_loader import load_tsv, load_ground_truth, print_summary_statistics
from src.normalization import normalize_dataframe
from src.blocking import CandidateBlocker
from src.features import FeatureExtractor
from src.labeling import assign_labels_and_sample_negatives
def run_sample_test():
    print("==================================================")
    print("      STARTING SMALL SAFE PIPELINE TEST          ")
    print("==================================================")
    print("\n[Stage 1] Loading sample subset (1,000 rows per source)...")
    s1_raw = load_tsv(TRAIN_SOURCE1).head(1000)
    s2_raw = load_tsv(TRAIN_SOURCE2).head(1000)
    s3_raw = load_tsv(TRAIN_SOURCE3).head(1000)
    gt_full = load_ground_truth(TRAIN_GROUND_TRUTH)
    sampled_s1_ids = set(s1_raw["entity_id"].values)
    gt_sampled = {s1_id: gt_full.get(s1_id, set()) for s1_id in sampled_s1_ids}
    print(f"Stage 1 Result: Input S1 rows={len(s1_raw):,}, S2 rows={len(s2_raw):,}, S3 rows={len(s3_raw):,}")
    print(f"Stage 1 Result: Ground truth entries for sampled S1={len(gt_sampled):,}")
    print("\n[Stage 2] Schema Validation...")
    required_cols = {"entity_id", "business_name", "business_address", "country"}
    for name, df in [("Source 1", s1_raw), ("Source 2", s2_raw), ("Source 3", s3_raw)]:
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Schema Error: {name} is missing columns: {missing_cols}")
    print("Stage 2 Result: Schema validation passed. All required columns present.")
    print("\n[Stage 3] Text Normalization...")
    norm_s1 = normalize_dataframe(s1_raw)
    s2_copy = s2_raw.copy()
    s2_copy["source"] = "S2"
    s3_copy = s3_raw.copy()
    s3_copy["source"] = "S3"
    s23_raw = pd.concat([s2_copy, s3_copy], ignore_index=True)
    norm_s23 = normalize_dataframe(s23_raw)
    print(f"Stage 3 Result: Norm S1 output rows={len(norm_s1):,}, Norm S23 output rows={len(norm_s23):,}")
    print("\n[Stage 4] Candidate Blocking Generation...")
    blocker = CandidateBlocker()
    candidates_df = blocker.generate_candidates(norm_s1, norm_s23)
    n_candidates = len(candidates_df)
    n_possible_cartesian = len(norm_s1) * len(norm_s23)
    reduction_ratio = (1.0 - n_candidates / n_possible_cartesian) * 100.0 if n_possible_cartesian > 0 else 0
    pair_tuples = list(zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]))
    n_unique_tuples = len(set(pair_tuples))
    has_duplicates = (n_unique_tuples != n_candidates)
    s23_id_set = set(norm_s23["entity_id"])
    cand_ids_valid = all(cid in s23_id_set for cid in candidates_df["candidate_entity_id"])
    print(f"Stage 4 Result: Generated candidate pairs={n_candidates:,} (Cartesian limit={n_possible_cartesian:,}, Search space reduction={reduction_ratio:.2f}%)")
    print(f"Stage 4 Verification: Duplicate pairs present={has_duplicates} (Unique={n_unique_tuples:,})")
    print(f"Stage 4 Verification: Candidates belong strictly to S2/S3={cand_ids_valid}")
    print("\n[Stage 5] Candidate Feature Generation...")
    feature_extractor = FeatureExtractor()
    feature_extractor.fit_tfidf_models(norm_s1, norm_s23)
    X_features, feature_names = feature_extractor.compute_features(candidates_df, norm_s1, norm_s23)
    has_nan = X_features.isna().any().any()
    has_inf = np.isinf(X_features.values).any()
    all_numeric_finite = (not has_nan) and (not has_inf)
    print(f"Stage 5 Result: Feature matrix shape={X_features.shape}, Total features={len(feature_names)}")
    print(f"Stage 5 Verification: All features finite numeric values (No NaN/Inf)={all_numeric_finite}")
    if has_nan or has_inf:
        nan_cols = X_features.columns[X_features.isna().any()].tolist()
        print(f"Stage 5 ERROR: Columns with NaN values: {nan_cols}")
    print("\n[Stage 6] Ground Truth Label Creation & Negative Sampling...")
    labeled_df = assign_labels_and_sample_negatives(candidates_df, gt_sampled)
    n_positives = (labeled_df["label"] == 1).sum()
    n_negatives = (labeled_df["label"] == 0).sum()
    empty_gt_s1 = [s1_id for s1_id, m in gt_sampled.items() if len(m) == 0]
    print(f"Stage 6 Result: Sampled S1 with empty GT={len(empty_gt_s1):,}")
    print(f"Stage 6 Result: Labeled output dataset rows={len(labeled_df):,}")
    print(f"Stage 6 Result: Positive pairs={n_positives:,}, Negative pairs={n_negatives:,}")
    print("\n==================================================")
    print("      SMALL PIPELINE TEST COMPLETED               ")
    print("==================================================")
    return True
if __name__ == "__main__":
    run_sample_test()
