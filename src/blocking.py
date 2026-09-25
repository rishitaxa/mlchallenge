"""
Blocking and candidate generation module for large-scale Business Entity Resolution.
Combines 5 complementary blocking strategies using UNION to maximize candidate recall.
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

from src.config import (
    NAME_PREFIX_LEN, MIN_TOKEN_LEN, MAX_TOKEN_FREQ_RATIO,
    TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_TOP_K,
    CANDIDATE_PAIRS_PATH, OUTPUT_DIR
)


class CandidateBlocker:
    """
    Candidate generation engine implementing 5 complementary blocking strategies:
    1. Same country + normalized business-name prefix
    2. Same country + business-name token overlap
    3. Same country + address token overlap
    4. Shared numeric/address tokens
    5. Character n-gram TF-IDF cosine similarity candidate retrieval
    """
    def __init__(
        self,
        prefix_len: int = NAME_PREFIX_LEN,
        min_token_len: int = MIN_TOKEN_LEN,
        max_token_freq_ratio: float = MAX_TOKEN_FREQ_RATIO,
        tfidf_ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
        tfidf_min_df: int = TFIDF_MIN_DF,
        tfidf_top_k: int = TFIDF_TOP_K
    ):
        self.prefix_len = prefix_len
        self.min_token_len = min_token_len
        self.max_token_freq_ratio = max_token_freq_ratio
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_top_k = tfidf_top_k

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s23_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate candidate pairs between Source 1 entities and Source 2/3 entities.
        Returns a DataFrame of candidate pairs with rule flags:
        source1_entity_id, candidate_entity_id, rule_prefix, rule_name_tok, rule_addr_tok, rule_num_tok, rule_tfidf
        """
        print("[Blocking] Starting 5-strategy Candidate Generation...")
        
        # Mapping index to entity_id for ultra-fast lookup
        s1_ids = s1_df["entity_id"].values
        s23_ids = s23_df["entity_id"].values
        
        # Candidate accumulator: dict mapping (s1_idx, s23_idx) -> bitmask/flags of triggered rules
        # Rule bits: 1: prefix, 2: name_tok, 4: addr_tok, 8: num_tok, 16: tfidf
        pair_flags: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Group indices by normalized country
        s1_country_map = defaultdict(list)
        for idx, country in enumerate(s1_df["norm_country"].values):
            s1_country_map[country].append(idx)
            
        s23_country_map = defaultdict(list)
        for idx, country in enumerate(s23_df["norm_country"].values):
            s23_country_map[country].append(idx)
            
        all_countries = set(s1_country_map.keys()).union(set(s23_country_map.keys()))
        print(f"[Blocking] Grouped data across {len(all_countries):,} distinct country keys.")

        # --- STRATEGIES 1, 2, 3, 4: Country-bucketed inverted indexes ---
        for country in tqdm(all_countries, desc="[Blocking] Rule 1-4 Indexing"):
            s1_indices = s1_country_map.get(country, [])
            s23_indices = s23_country_map.get(country, [])
            
            if not s1_indices or not s23_indices:
                continue
                
            n_s23 = len(s23_indices)
            freq_threshold = max(5, int(n_s23 * self.max_token_freq_ratio))
            
            # Build inverted indexes for S2/3 candidates in this country
            prefix_index = defaultdict(list)
            name_tok_index = defaultdict(list)
            addr_tok_index = defaultdict(list)
            num_tok_index = defaultdict(list)
            
            # Populate indexes from s23 entities
            for s23_idx in s23_indices:
                norm_name = s23_df.at[s23_idx, "norm_name"]
                name_tokens = s23_df.at[s23_idx, "name_tokens"]
                addr_tokens = s23_df.at[s23_idx, "address_tokens"]
                num_tokens = s23_df.at[s23_idx, "numeric_tokens"]
                
                # Rule 1: Name Prefix
                if len(norm_name) >= self.prefix_len:
                    prefix = norm_name[:self.prefix_len]
                    prefix_index[prefix].append(s23_idx)
                    
                # Rule 2: Name Token
                for tok in name_tokens:
                    if len(tok) >= self.min_token_len:
                        name_tok_index[tok].append(s23_idx)
                        
                # Rule 3: Address Token
                for tok in addr_tokens:
                    if len(tok) >= self.min_token_len:
                        addr_tok_index[tok].append(s23_idx)
                        
                # Rule 4: Shared Numeric Token
                for num_tok in num_tokens:
                    num_tok_index[num_tok].append(s23_idx)

            # Filter high frequency tokens (stopwords) from token indexes to prevent explosion
            filtered_name_tok_index = {
                t: idxs for t, idxs in name_tok_index.items() if len(idxs) <= freq_threshold
            }
            filtered_addr_tok_index = {
                t: idxs for t, idxs in addr_tok_index.items() if len(idxs) <= freq_threshold
            }

            # Query inverted indexes for each S1 entity in this country
            for s1_idx in s1_indices:
                norm_name = s1_df.at[s1_idx, "norm_name"]
                name_tokens = s1_df.at[s1_idx, "name_tokens"]
                addr_tokens = s1_df.at[s1_idx, "address_tokens"]
                num_tokens = s1_df.at[s1_idx, "numeric_tokens"]
                
                # Rule 1: Prefix match
                if len(norm_name) >= self.prefix_len:
                    prefix = norm_name[:self.prefix_len]
                    for cand_idx in prefix_index.get(prefix, []):
                        pair_flags[(s1_idx, cand_idx)] |= 1
                        
                # Rule 2: Name token match
                for tok in name_tokens:
                    if tok in filtered_name_tok_index:
                        for cand_idx in filtered_name_tok_index[tok]:
                            pair_flags[(s1_idx, cand_idx)] |= 2
                            
                # Rule 3: Address token match
                for tok in addr_tokens:
                    if tok in filtered_addr_tok_index:
                        for cand_idx in filtered_addr_tok_index[tok]:
                            pair_flags[(s1_idx, cand_idx)] |= 4
                            
                # Rule 4: Shared numeric token match
                for num_tok in num_tokens:
                    if num_tok in num_tok_index:
                        for cand_idx in num_tok_index[num_tok]:
                            pair_flags[(s1_idx, cand_idx)] |= 8

        # --- STRATEGY 5: Character n-gram TF-IDF Candidate Retrieval ---
        print("[Blocking] Executing Strategy 5: Character n-gram TF-IDF Retrieval...")
        s1_text = (s1_df["norm_name"] + " " + s1_df["norm_address"]).tolist()
        s23_text = (s23_df["norm_name"] + " " + s23_df["norm_address"]).tolist()
        
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.tfidf_ngram_range,
            min_df=self.tfidf_min_df,
            dtype=np.float32
        )
        
        # Fit vectorizer on target pool s23 to save memory and project s1
        s23_tfidf = vectorizer.fit_transform(s23_text)
        s1_tfidf = vectorizer.transform(s1_text)
        
        # Perform batched sparse dot product multiplication to find top-K per S1 row
        batch_size = 5000
        n_s1 = s1_tfidf.shape[0]
        
        for start_idx in tqdm(range(0, n_s1, batch_size), desc="[Blocking] TF-IDF Dot Product"):
            end_idx = min(start_idx + batch_size, n_s1)
            batch_s1_tfidf = s1_tfidf[start_idx:end_idx]
            
            # Similarity matrix batch: (batch_size x N_s23)
            sim_batch = batch_s1_tfidf.dot(s23_tfidf.T)
            
            # Efficient top-K selection per row in sparse batch
            for row_offset in range(end_idx - start_idx):
                s1_idx = start_idx + row_offset
                row_data = sim_batch[row_offset]
                
                if row_data.nnz == 0:
                    continue
                    
                # Extract non-zero column indices and values
                cand_indices = row_data.indices
                cand_sims = row_data.data
                
                if len(cand_sims) > self.tfidf_top_k:
                    # Argpartition to get top K indices efficiently
                    top_k_arg = np.argpartition(cand_sims, -self.tfidf_top_k)[-self.tfidf_top_k:]
                    top_cand_indices = cand_indices[top_k_arg]
                else:
                    top_cand_indices = cand_indices
                    
                for cand_idx in top_cand_indices:
                    pair_flags[(s1_idx, cand_idx)] |= 16

        print(f"[Blocking] Candidate Generation complete. Total unique candidate pairs: {len(pair_flags):,}")
        
        # Memory-efficient columnar unpacking
        n_pairs = len(pair_flags)
        s1_idx_arr = np.zeros(n_pairs, dtype=np.int32)
        cand_idx_arr = np.zeros(n_pairs, dtype=np.int32)
        flag_arr = np.zeros(n_pairs, dtype=np.int32)
        
        for i, ((s1_idx, cand_idx), flag_val) in enumerate(pair_flags.items()):
            s1_idx_arr[i] = s1_idx
            cand_idx_arr[i] = cand_idx
            flag_arr[i] = flag_val
            
        rule_prefix = ((flag_arr & 1) > 0).astype(np.int8)
        rule_name_tok = ((flag_arr & 2) > 0).astype(np.int8)
        rule_addr_tok = ((flag_arr & 4) > 0).astype(np.int8)
        rule_num_tok = ((flag_arr & 8) > 0).astype(np.int8)
        rule_tfidf = ((flag_arr & 16) > 0).astype(np.int8)
        rule_count = (rule_prefix + rule_name_tok + rule_addr_tok + rule_num_tok + rule_tfidf).astype(np.int8)
        
        candidates_df = pd.DataFrame({
            "source1_entity_id": s1_ids[s1_idx_arr],
            "candidate_entity_id": s23_ids[cand_idx_arr],
            "s1_index": s1_idx_arr,
            "cand_index": cand_idx_arr,
            "rule_prefix": rule_prefix,
            "rule_name_tok": rule_name_tok,
            "rule_addr_tok": rule_addr_tok,
            "rule_num_tok": rule_num_tok,
            "rule_tfidf": rule_tfidf,
            "rule_count": rule_count
        })
        return candidates_df


def save_candidate_pairs_tsv(
    candidates_df: pd.DataFrame,
    all_s1_ids: List[str],
    output_path: str = CANDIDATE_PAIRS_PATH
) -> None:
    """
    Save candidate pairs TSV matching requirement format:
    source1_entity_id \t candidate_entity_ids (space-separated)
    Guarantees every S1 entity appears in output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Fast dict lookup mapping s1_id -> set of candidate IDs
    grouped: Dict[str, Set[str]] = defaultdict(set)
    if len(candidates_df) > 0:
        for s1_id, cand_id in zip(candidates_df["source1_entity_id"].values, candidates_df["candidate_entity_id"].values):
            grouped[s1_id].add(cand_id)
            
    rows = []
    for s1_id in all_s1_ids:
        cand_set = grouped.get(s1_id, set())
        cand_str = " ".join(sorted(cand_set)) if cand_set else ""
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cand_str})
        
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, sep="\t", index=False)
    print(f"[Blocking] Saved output candidate pairs file to: {output_path}")
