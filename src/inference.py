import os
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from src.config import MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, PAIR_BATCH_SIZE
from src.data_loader import load_test_data
from src.normalization import normalize_dataframe
from src.blocking import CandidateBlocker, save_candidate_pairs_tsv
from src.features import FeatureExtractor
from src.model import EntityResolutionModel
def run_inference_on_test(
    model: EntityResolutionModel,
    feature_extractor: FeatureExtractor,
    threshold: float,
    batch_size: int = PAIR_BATCH_SIZE
) -> Tuple[str, str]:
    print("\n========================================")
    print("        STARTING TEST INFERENCE         ")
    print("========================================")
    test_s1, test_s2, test_s3 = load_test_data()
    print("[Inference] Normalizing test dataset records...")
    norm_test_s1 = normalize_dataframe(test_s1)
    test_s2_copy = test_s2.copy()
    test_s2_copy["source"] = "S2"
    test_s3_copy = test_s3.copy()
    test_s3_copy["source"] = "S3"
    test_s23 = pd.concat([test_s2_copy, test_s3_copy], ignore_index=True)
    norm_test_s23 = normalize_dataframe(test_s23)
    blocker = CandidateBlocker()
    test_candidates_df = blocker.generate_candidates(norm_test_s1, norm_test_s23)
    all_s1_test_ids = list(norm_test_s1["entity_id"].values)
    save_candidate_pairs_tsv(test_candidates_df, all_s1_test_ids, output_path=CANDIDATE_PAIRS_PATH)
    if len(test_candidates_df) == 0:
        print("[Inference] WARNING: No candidates generated for test set!")
        save_matching_results([], all_s1_test_ids, MATCHING_RESULTS_PATH)
        return MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH
    X_test, _ = feature_extractor.compute_features(
        test_candidates_df,
        norm_test_s1,
        norm_test_s23,
        batch_size=batch_size
    )
    print(f"[Inference] Predicting match probabilities for {len(X_test):,} candidate pairs...")
    test_probs = model.predict_proba(X_test)
    test_candidates_df["match_prob"] = test_probs
    print(f"[Inference] Applying decision threshold: {threshold:.2f}")
    matched_pairs_df = test_candidates_df[test_candidates_df["match_prob"] >= threshold]
    print(f"[Inference] Selected {len(matched_pairs_df):,} matching pairs passing threshold {threshold:.2f}")
    save_matching_results(matched_pairs_df, all_s1_test_ids, MATCHING_RESULTS_PATH)
    s23_test_ids = set(norm_test_s23["entity_id"].values)
    validate_output_files(MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH, all_s1_test_ids, s23_test_ids)
    print("========================================")
    print("   TEST INFERENCE COMPLETED SUCCESSFULLY ")
    print("========================================")
    print(f"Output 1 (Matching Results): {MATCHING_RESULTS_PATH}")
    print(f"Output 2 (Candidate Pairs): {CANDIDATE_PAIRS_PATH}\n")
    return MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH
def save_matching_results(
    matched_pairs_df: pd.DataFrame,
    all_s1_ids: List[str],
    output_path: str = MATCHING_RESULTS_PATH
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped: Dict[str, Set[str]] = {}
    if len(matched_pairs_df) > 0:
        for s1_id, cand_id in zip(matched_pairs_df["source1_entity_id"].values, matched_pairs_df["candidate_entity_id"].values):
            if s1_id not in grouped:
                grouped[s1_id] = set()
            grouped[s1_id].add(cand_id)
    rows = []
    for s1_id in all_s1_ids:
        match_set = grouped.get(s1_id, set())
        match_str = " ".join(sorted(match_set)) if match_set else ""
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": match_str
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"[Inference] Saved output matching results file to: {output_path} ({len(out_df):,} rows)")
def validate_output_files(
    matching_path: str,
    candidate_path: str,
    all_s1_ids: List[str],
    s23_ids_set: Set[str]
) -> Dict[str, int]:
    print("\n--- OUTPUT FORMAT & METRIC VALIDATION ---")
    res_df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str, keep_default_na=False)
    expected_cols = ["source1_entity_id", "matched_entity_ids"]
    assert list(res_df.columns) == expected_cols, f"Invalid columns in {matching_path}: {list(res_df.columns)}"
    print("Validation Check 1: Columns match exactly ['source1_entity_id', 'matched_entity_ids'] -> PASSED")
    assert len(res_df) == len(all_s1_ids), f"Row count mismatch: expected {len(all_s1_ids)}, got {len(res_df)}"
    assert not res_df["source1_entity_id"].duplicated().any(), "Duplicate Source 1 IDs found in output!"
    print(f"Validation Check 2: Exactly one row per test S1 entity ({len(res_df):,} rows), zero duplicates -> PASSED")
    cand_map = {}
    for _, row in cand_df.iterrows():
        cands = set(row["candidate_entity_ids"].split()) if row["candidate_entity_ids"] else set()
        cand_map[row["source1_entity_id"]] = cands
    s1_set = set(all_s1_ids)
    zero_matches, one_match, multi_matches = 0, 0, 0
    total_matches = 0
    for _, row in res_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        matched_list = matched_str.split() if matched_str else []
        n_m = len(matched_list)
        total_matches += n_m
        if n_m == 0:
            zero_matches += 1
        elif n_m == 1:
            one_match += 1
        else:
            multi_matches += 1
        for m_id in matched_list:
            assert m_id not in s1_set, f"Error: Source 1 ID {m_id} erroneously listed as matched entity!"
            assert m_id in s23_ids_set, f"Error: Matched ID {m_id} does not belong to S2/S3 candidate pool!"
            assert m_id in cand_map.get(s1_id, set()), f"Error: Final match {m_id} for {s1_id} missing from candidate_pairs.tsv!"
    print("Validation Check 3: No S1 ID listed as match -> PASSED")
    print("Validation Check 4: All matched IDs belong to S2/S3 pool -> PASSED")
    print("Validation Check 5: Every final match exists in candidate_pairs.tsv -> PASSED")
    print("\n--- TEST MATCH DISTRIBUTION BREAKDOWN ---")
    print(f"Total S1 test entities:              {len(res_df):,}")
    print(f"Entities with ZERO matches:           {zero_matches:,}")
    print(f"Entities with EXACTLY ONE match:      {one_match:,}")
    print(f"Entities with MULTIPLE matches:       {multi_matches:,}")
    print(f"Total final predicted match pairs:    {total_matches:,}")
    print("-----------------------------------------\n")
    return {
        "zero_matches": zero_matches,
        "one_match": one_match,
        "multi_matches": multi_matches,
        "total_matches": total_matches
    }
