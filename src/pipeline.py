import os
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from src.config import (
    VAL_SIZE, RANDOM_SEED, PAIR_BATCH_SIZE, THRESHOLDS, BETA,
    MODEL_PATH, MODEL_METADATA_PATH, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH
)
from src.data_loader import load_train_data
from src.normalization import normalize_dataframe
from src.blocking import CandidateBlocker
from src.labeling import assign_labels_and_sample_negatives
from src.features import FeatureExtractor
from src.model import EntityResolutionModel
from src.evaluation import optimize_threshold, evaluate_entity_level_resolution
from src.inference import run_inference_on_test
def run_pipeline(
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED,
    batch_size: int = PAIR_BATCH_SIZE,
    skip_test_inference: bool = False
) -> Tuple[EntityResolutionModel, float]:
    print("==================================================")
    print("  BUSINESS ENTITY RESOLUTION PIPELINE STARTING    ")
    print("==================================================")
    np.random.seed(random_seed)
    print("\n--- STEP 1: DATA LOADING ---")
    s1_df, s2_df, s3_df, ground_truth = load_train_data()
    print("\n--- STEP 2: NORMALIZATION ---")
    print("[Pipeline] Normalizing Source 1 records...")
    norm_s1 = normalize_dataframe(s1_df)
    print("[Pipeline] Combining & Normalizing Source 2 and Source 3 records...")
    s2_copy = s2_df.copy()
    s2_copy["source"] = "S2"
    s3_copy = s3_df.copy()
    s3_copy["source"] = "S3"
    s23_df = pd.concat([s2_copy, s3_copy], ignore_index=True)
    norm_s23 = normalize_dataframe(s23_df)
    print("\n--- STEP 3: SOURCE 1 LEVEL VALIDATION SPLIT ---")
    all_s1_ids = norm_s1["entity_id"].values
    n_s1 = len(all_s1_ids)
    shuffled_s1_ids = np.random.choice(all_s1_ids, size=n_s1, replace=False)
    n_val = int(n_s1 * val_size)
    val_s1_id_set = set(shuffled_s1_ids[:n_val])
    train_s1_id_set = set(shuffled_s1_ids[n_val:])
    train_s1 = norm_s1[norm_s1["entity_id"].isin(train_s1_id_set)].reset_index(drop=True)
    val_s1 = norm_s1[norm_s1["entity_id"].isin(val_s1_id_set)].reset_index(drop=True)
    print(f"[Pipeline] Source 1 Split: {len(train_s1):,} train entities ({1-val_size:.0%}), {len(val_s1):,} val entities ({val_size:.0%}).")
    print(f"[Pipeline] Validation leakage check: Train & Val S1 intersection size = {len(train_s1_id_set.intersection(val_s1_id_set))}")
    print("\n--- STEP 4: BLOCKING & CANDIDATE GENERATION ---")
    blocker = CandidateBlocker()
    print("[Pipeline] Generating candidates for TRAIN set...")
    train_candidates = blocker.generate_candidates(train_s1, norm_s23)
    print("[Pipeline] Generating candidates for VALIDATION set...")
    val_candidates = blocker.generate_candidates(val_s1, norm_s23)
    print("\n--- STEP 5: LABEL ASSIGNMENT & NEGATIVE SAMPLING ---")
    train_labeled = assign_labels_and_sample_negatives(train_candidates, ground_truth, random_seed=random_seed)
    val_labeled = assign_labels_and_sample_negatives(val_candidates, ground_truth, random_seed=random_seed)
    print("\n--- STEP 6: FEATURE ENGINEERING ---")
    feature_extractor = FeatureExtractor()
    feature_extractor.fit_tfidf_models(norm_s1, norm_s23)
    print("[Pipeline] Computing features for TRAIN set...")
    X_train, feature_names = feature_extractor.compute_features(train_labeled, norm_s1, norm_s23, batch_size=batch_size)
    y_train = train_labeled["label"].values
    print("[Pipeline] Computing features for VALIDATION set...")
    X_val, _ = feature_extractor.compute_features(val_labeled, norm_s1, norm_s23, batch_size=batch_size)
    y_val = val_labeled["label"].values
    print("\n--- STEP 7: MODEL TRAINING ---")
    model = EntityResolutionModel(random_seed=random_seed)
    model.fit(X_train, y_train, X_val, y_val)
    print("\n--- STEP 8: THRESHOLD OPTIMIZATION ---")
    val_probs = model.predict_proba(X_val)
    best_thresh, opt_res_df = optimize_threshold(y_val, val_probs, thresholds=THRESHOLDS, beta=BETA)
    model.best_threshold = best_thresh
    val_gt_subset = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1_id_set}
    evaluate_entity_level_resolution(val_labeled, val_probs, val_gt_subset, threshold=best_thresh, beta=BETA)
    print("\n--- STEP 9: SAVING TRAINED MODEL ---")
    model.save(MODEL_PATH, MODEL_METADATA_PATH)
    if not skip_test_inference:
        print("\n--- STEP 10: TEST INFERENCE & OUTPUT GENERATION ---")
        run_inference_on_test(model, feature_extractor, best_thresh, batch_size=batch_size)
    print("\n==================================================")
    print("  BUSINESS ENTITY RESOLUTION PIPELINE COMPLETE!   ")
    print("==================================================")
    return model, best_thresh
