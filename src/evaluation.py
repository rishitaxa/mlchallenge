import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple
from src.config import BETA, THRESHOLDS
def calculate_fbeta(precision: float, recall: float, beta: float = BETA) -> float:
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    return numerator / denominator if denominator > 0 else 0.0
def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = BETA
) -> Dict[str, float]:
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_score = calculate_fbeta(precision, recall, beta=beta)
    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "num_predictions": int(tp + fp)
    }
def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: List[float] = THRESHOLDS,
    beta: float = BETA
) -> Tuple[float, pd.DataFrame]:
    results = []
    best_threshold = thresholds[0]
    best_f_score = -1.0
    print("\n========================================")
    print("      THRESHOLD OPTIMIZATION (F0.5)     ")
    print("========================================")
    print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F0.5':<12}{'Num Preds':<12}")
    print("-" * 60)
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        metrics = evaluate_predictions(y_true, y_pred, beta=beta)
        prec = metrics["precision"]
        rec = metrics["recall"]
        f_val = metrics["f_score"]
        n_preds = metrics["num_predictions"]
        results.append({
            "threshold": thresh,
            "precision": prec,
            "recall": rec,
            "f_0.5": f_val,
            "num_predictions": n_preds
        })
        print(f"{thresh:<12.2f}{prec:<12.4f}{rec:<12.4f}{f_val:<12.4f}{n_preds:<12,}")
        if f_val > best_f_score:
            best_f_score = f_val
            best_threshold = thresh
    print("-" * 60)
    print(f"Optimal Threshold Selected: {best_threshold:.2f} (Validation F0.5: {best_f_score:.4f})")
    print("========================================\n")
    res_df = pd.DataFrame(results)
    return best_threshold, res_df
def evaluate_entity_level_resolution(
    candidates_df: pd.DataFrame,
    y_probs: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    threshold: float,
    beta: float = BETA
) -> Dict[str, float]:
    df = candidates_df.copy()
    df["prob"] = y_probs
    df_match = df[df["prob"] >= threshold]
    pred_matches_map = df_match.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: set(ids)
    ).to_dict()
    all_s1_ids = list(ground_truth.keys())
    total_tp, total_fp, total_fn = 0, 0, 0
    for s1_id in all_s1_ids:
        gt_set = ground_truth.get(s1_id, set())
        pred_set = pred_matches_map.get(s1_id, set())
        tp = len(gt_set.intersection(pred_set))
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f05 = calculate_fbeta(precision, recall, beta=beta)
    print("\n--- Final Validation Entity-Level Metrics ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F0.5 Score: {f05:.4f}")
    print(f"True Positives: {total_tp:,}, False Positives: {total_fp:,}, False Negatives: {total_fn:,}")
    print("---------------------------------------------")
    return {
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    }
