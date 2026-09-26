import numpy as np
import pandas as pd
from typing import Dict, Set, Tuple
from src.config import RANDOM_SEED, NEGATIVE_RATIO, HARD_NEGATIVE_RATIO
def assign_labels_and_sample_negatives(
    candidates_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    negative_ratio: float = NEGATIVE_RATIO,
    hard_ratio: float = HARD_NEGATIVE_RATIO,
    random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    np.random.seed(random_seed)
    print("[Labeling] Assigning labels to candidate pairs...")
    s1_vals = candidates_df["source1_entity_id"].values
    cand_vals = candidates_df["candidate_entity_id"].values
    labels = [
        int(cand_id in ground_truth.get(s1_id, set()))
        for s1_id, cand_id in zip(s1_vals, cand_vals)
    ]
    df = candidates_df.copy()
    df["label"] = np.array(labels, dtype=np.int32)
    pos_mask = (df["label"] == 1)
    n_positives = pos_mask.sum()
    n_total_candidates = len(df)
    total_gt_pairs = sum(len(matches) for matches in ground_truth.values())
    recall_pct = (n_positives / total_gt_pairs * 100.0) if total_gt_pairs > 0 else 0.0
    print(f"[Labeling] Positives found in candidate set: {n_positives:,} / {total_gt_pairs:,} GT matches ({recall_pct:.2f}% Candidate Recall).")
    print(f"[Labeling] Candidate set breakdown: {n_positives:,} positives, {n_total_candidates - n_positives:,} negatives.")
    if n_positives == 0:
        print("[Labeling] WARNING: 0 positive pairs found in candidates!")
        return df
    desired_negatives = int(n_positives * negative_ratio)
    negatives_df = df[~pos_mask].copy()
    if len(negatives_df) <= desired_negatives:
        print(f"[Labeling] Total negatives ({len(negatives_df):,}) <= desired limit ({desired_negatives:,}). Keeping all negatives.")
        sampled_df = df.copy()
    else:
        negatives_df["hardness_score"] = (
            negatives_df["rule_count"] * 2.0 +
            negatives_df["rule_tfidf"] * 3.0 +
            negatives_df["rule_prefix"] * 1.5 +
            negatives_df["rule_name_tok"] * 1.0
        )
        n_hard = int(desired_negatives * hard_ratio)
        n_random = desired_negatives - n_hard
        hard_negatives = negatives_df.nlargest(n_hard, "hardness_score")
        remaining_negatives = negatives_df.drop(hard_negatives.index)
        if len(remaining_negatives) >= n_random:
            random_negatives = remaining_negatives.sample(n=n_random, random_state=random_seed)
        else:
            random_negatives = remaining_negatives
        sampled_negatives = pd.concat([hard_negatives, random_negatives], ignore_index=True)
        sampled_df = pd.concat([df[pos_mask], sampled_negatives], ignore_index=True)
        if "hardness_score" in sampled_df.columns:
            sampled_df = sampled_df.drop(columns=["hardness_score"])
    sampled_df = sampled_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    print(f"[Labeling] Final sampled dataset size: {len(sampled_df):,} candidate pairs ({n_positives:,} pos, {len(sampled_df) - n_positives:,} neg).")
    return sampled_df
