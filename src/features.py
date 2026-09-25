"""
Feature engineering module for candidate pairs.
Computes string, token, TF-IDF cosine, and structural similarity features in efficient batches.
"""
import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple
from tqdm import tqdm
from src.config import PAIR_BATCH_SIZE


# --- Pure Python Fast String Distance Algorithms ---

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein distance between two strings."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
        
    m, n = len(s1), len(s2)
    # Ensure s1 is smaller to minimize array size
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
        
    dp = list(range(m + 1))
    for j, c2 in enumerate(s2, 1):
        new_dp = [j] + [0] * m
        for i, c1 in enumerate(s1, 1):
            cost = 0 if c1 == c2 else 1
            new_dp[i] = min(dp[i] + 1, new_dp[i - 1] + 1, dp[i - 1] + cost)
        dp = new_dp
    return dp[m]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Compute normalized Levenshtein similarity score in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = compute_levenshtein_distance(s1, s2)
    return 1.0 - (dist / max_len)


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity score in [0.0, 1.0]."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
        
    match_distance = (max(len1, len2) // 2) - 1
    if match_distance < 0:
        match_distance = 0
        
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
            
    if matches == 0:
        return 0.0
        
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
        
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2.0) / matches) / 3.0
    
    # Winkler adjustment for common prefix up to 4 chars
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
            
    return jaro + prefix * p * (1.0 - jaro)


def token_jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity of token lists."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    set1, set2 = set(tokens1), set(tokens2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def token_set_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token set similarity (intersection over min token set size)."""
    if not tokens1 and not tokens2:
        return 1.0
    if not tokens1 or not tokens2:
        return 0.0
    set1, set2 = set(tokens1), set(tokens2)
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    return intersection / min_len if min_len > 0 else 0.0


def token_overlap_count(tokens1: List[str], tokens2: List[str]) -> int:
    """Compute raw token intersection count."""
    if not tokens1 or not tokens2:
        return 0
    return len(set(tokens1).intersection(set(tokens2)))


def token_sort_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute similarity of sorted token strings."""
    s1 = " ".join(sorted(tokens1))
    s2 = " ".join(sorted(tokens2))
    return levenshtein_similarity(s1, s2)


def postal_code_similarity(postal1: str, postal2: str) -> float:
    """Compare extracted postal/PIN codes."""
    if not postal1 or not postal2:
        return 0.5  # Neutral indicator when postal code is missing
    if postal1 == postal2:
        return 1.0
    return levenshtein_similarity(postal1, postal2)


class FeatureExtractor:
    """
    Batched feature engineering engine for candidate pairs.
    Pre-computes sparse TF-IDF representations and computes rich pairwise features.
    """
    def __init__(self):
        self.fitted = False
        self.name_char_tfidf = None
        self.name_word_tfidf = None
        self.addr_char_tfidf = None
        self.addr_word_tfidf = None
        
        # Pre-transformed sparse matrices for S1 and S23 entities
        self.s1_name_char_mat = None
        self.s1_name_word_mat = None
        self.s1_addr_char_mat = None
        self.s1_addr_word_mat = None
        
        self.s23_name_char_mat = None
        self.s23_name_word_mat = None
        self.s23_addr_char_mat = None
        self.s23_addr_word_mat = None

    def fit_tfidf_models(self, s1_df: pd.DataFrame, s23_df: pd.DataFrame) -> None:
        """Fit character & word TF-IDF vectorizers on S1 and S23 text corpus."""
        print("[Features] Fitting TF-IDF Vectorizers...")
        
        all_names = pd.concat([s1_df["norm_name"], s23_df["norm_name"]], ignore_index=True)
        all_addrs = pd.concat([s1_df["norm_address"], s23_df["norm_address"]], ignore_index=True)
        
        self.name_char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, dtype=np.float32)
        self.name_word_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, dtype=np.float32)
        self.addr_char_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, dtype=np.float32)
        self.addr_word_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, dtype=np.float32)
        
        self.name_char_tfidf.fit(all_names)
        self.name_word_tfidf.fit(all_names)
        self.addr_char_tfidf.fit(all_addrs)
        self.addr_word_tfidf.fit(all_addrs)
        
        print("[Features] Pre-transforming sparse TF-IDF matrices for entities...")
        self.s1_name_char_mat = self.name_char_tfidf.transform(s1_df["norm_name"])
        self.s1_name_word_mat = self.name_word_tfidf.transform(s1_df["norm_name"])
        self.s1_addr_char_mat = self.addr_char_tfidf.transform(s1_df["norm_address"])
        self.s1_addr_word_mat = self.addr_word_tfidf.transform(s1_df["norm_address"])
        
        self.s23_name_char_mat = self.name_char_tfidf.transform(s23_df["norm_name"])
        self.s23_name_word_mat = self.name_word_tfidf.transform(s23_df["norm_name"])
        self.s23_addr_char_mat = self.addr_char_tfidf.transform(s23_df["norm_address"])
        self.s23_addr_word_mat = self.addr_word_tfidf.transform(s23_df["norm_address"])
        
        self.fitted = True

    def compute_features(
        self,
        candidates_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame,
        batch_size: int = PAIR_BATCH_SIZE
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract feature matrix X for candidate pairs in memory-efficient batches.
        Returns:
          (feature_df, feature_names_list)
        """
        if not self.fitted:
            self.fit_tfidf_models(s1_df, s23_df)
            
        print(f"[Features] Generating features for {len(candidates_df):,} candidate pairs in batches of {batch_size:,}...")
        
        s1_indices = candidates_df["s1_index"].values
        cand_indices = candidates_df["cand_index"].values
        
        # Precompute vector sparse row multiplication for TF-IDF cosine features across all candidate pairs
        print("[Features] Computing sparse TF-IDF Cosine similarities...")
        
        # Dot product element-wise row sum: (s1_mat[r1].multiply(s23_mat[r2])).sum(axis=1)
        name_char_sims = np.array(self.s1_name_char_mat[s1_indices].multiply(self.s23_name_char_mat[cand_indices]).sum(axis=1)).ravel()
        name_word_sims = np.array(self.s1_name_word_mat[s1_indices].multiply(self.s23_name_word_mat[cand_indices]).sum(axis=1)).ravel()
        addr_char_sims = np.array(self.s1_addr_char_mat[s1_indices].multiply(self.s23_addr_char_mat[cand_indices]).sum(axis=1)).ravel()
        addr_word_sims = np.array(self.s1_addr_word_mat[s1_indices].multiply(self.s23_addr_word_mat[cand_indices]).sum(axis=1)).ravel()
        
        # Fast lookup arrays for attributes
        s1_names = s1_df["norm_name"].values
        s1_addrs = s1_df["norm_address"].values
        s1_countries = s1_df["norm_country"].values
        s1_name_toks = s1_df["name_tokens"].values
        s1_addr_toks = s1_df["address_tokens"].values
        s1_num_toks = s1_df["numeric_tokens"].values
        s1_postals = s1_df["postal_code"].values
        
        s23_names = s23_df["norm_name"].values
        s23_addrs = s23_df["norm_address"].values
        s23_countries = s23_df["norm_country"].values
        s23_name_toks = s23_df["name_tokens"].values
        s23_addr_toks = s23_df["address_tokens"].values
        s23_num_toks = s23_df["numeric_tokens"].values
        s23_postals = s23_df["postal_code"].values
        
        # Precompute candidate source indicators ('S2' -> 0, 'S3' -> 1)
        if "source" in s23_df.columns:
            s23_sources = (s23_df["source"] == "S3").astype(int).values
        else:
            s23_sources = np.zeros(len(s23_df), dtype=int)

        n_pairs = len(candidates_df)
        feature_chunks = []
        
        for start_i in tqdm(range(0, n_pairs, batch_size), desc="[Features] Pair Batch Extraction"):
            end_i = min(start_i + batch_size, n_pairs)
            
            b_s1_idx = s1_indices[start_i:end_i]
            b_cand_idx = cand_indices[start_i:end_i]
            
            chunk_rows = []
            for i_loc, (i_s1, i_cand) in enumerate(zip(b_s1_idx, b_cand_idx)):
                pair_idx = start_i + i_loc
                
                n1, n2 = s1_names[i_s1], s23_names[i_cand]
                a1, a2 = s1_addrs[i_s1], s23_addrs[i_cand]
                c1, c2 = s1_countries[i_s1], s23_countries[i_cand]
                
                nt1, nt2 = s1_name_toks[i_s1], s23_name_toks[i_cand]
                at1, at2 = s1_addr_toks[i_s1], s23_addr_toks[i_cand]
                num1, num2 = s1_num_toks[i_s1], s23_num_toks[i_cand]
                p1, p2 = s1_postals[i_s1], s23_postals[i_cand]
                
                # NAME FEATURES
                name_exact = int(n1 == n2 and len(n1) > 0)
                name_lev = levenshtein_similarity(n1, n2)
                name_jw = jaro_winkler_similarity(n1, n2)
                name_jaccard = token_jaccard_similarity(nt1, nt2)
                name_overlap = token_overlap_count(nt1, nt2)
                name_token_set = token_set_similarity(nt1, nt2)
                name_token_sort = token_sort_similarity(nt1, nt2)
                name_len_diff = abs(len(n1) - len(n2))
                
                # ADDRESS FEATURES
                addr_exact = int(a1 == a2 and len(a1) > 0)
                addr_lev = levenshtein_similarity(a1, a2)
                addr_jaccard = token_jaccard_similarity(at1, at2)
                addr_overlap = token_overlap_count(at1, at2)
                num_overlap = token_overlap_count(num1, num2)
                postal_sim = postal_code_similarity(p1, p2)
                addr_len_diff = abs(len(a1) - len(a2))
                
                # OTHER FEATURES
                c_exact = int(c1 == c2 and len(c1) > 0)
                c_mismatch = int(c1 != c2 and len(c1) > 0 and len(c2) > 0)
                combined_sim = (name_jw * 0.6 + addr_lev * 0.4)
                src_indicator = s23_sources[i_cand]
                shared_tokens = name_overlap + addr_overlap
                
                row_dict = {
                    # Name features
                    "name_exact_match": name_exact,
                    "name_levenshtein_sim": name_lev,
                    "name_jaro_winkler_sim": name_jw,
                    "name_token_jaccard": name_jaccard,
                    "name_token_overlap": name_overlap,
                    "name_token_set_sim": name_token_set,
                    "name_token_sort_sim": name_token_sort,
                    "name_char_tfidf_cosine": name_char_sims[pair_idx],
                    "name_word_tfidf_cosine": name_word_sims[pair_idx],
                    "name_length_diff": name_len_diff,
                    
                    # Address features
                    "address_exact_match": addr_exact,
                    "address_levenshtein_sim": addr_lev,
                    "address_token_jaccard": addr_jaccard,
                    "address_token_overlap": addr_overlap,
                    "address_char_tfidf_cosine": addr_char_sims[pair_idx],
                    "address_word_tfidf_cosine": addr_word_sims[pair_idx],
                    "numeric_token_overlap": num_overlap,
                    "postal_code_sim": postal_sim,
                    "address_length_diff": addr_len_diff,
                    
                    # Other features
                    "country_exact_match": c_exact,
                    "country_mismatch": c_mismatch,
                    "combined_name_address_sim": combined_sim,
                    "source_indicator": src_indicator,
                    "shared_token_counts": shared_tokens
                }
                chunk_rows.append(row_dict)
                
            feature_chunks.append(pd.DataFrame(chunk_rows))
            
        features_df = pd.concat(feature_chunks, ignore_index=True)
        
        # Attach blocking rule indicators if present in candidates_df
        blocking_cols = [col for col in candidates_df.columns if col.startswith("rule_")]
        for b_col in blocking_cols:
            features_df[b_col] = candidates_df[b_col].values
            
        feature_names = list(features_df.columns)
        print(f"[Features] Extracted {len(feature_names)} features for dataset.")
        return features_df, feature_names
