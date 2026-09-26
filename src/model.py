import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
HAS_LIGHTGBM = False
HAS_XGBOOST = False
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass
from sklearn.ensemble import HistGradientBoostingClassifier
from src.config import MODEL_PATH, MODEL_METADATA_PATH, RANDOM_SEED
class EntityResolutionModel:
    def __init__(self, random_seed: int = RANDOM_SEED):
        self.random_seed = random_seed
        self.model = None
        self.backend = None
        self.feature_names: List[str] = []
        self.best_threshold: float = 0.50
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "EntityResolutionModel":
        self.feature_names = list(X_train.columns)
        if HAS_LIGHTGBM:
            print("[Model] Initializing LightGBM Classifier backend...")
            self.backend = "lightgbm"
            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1
            )
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
            else:
                self.model.fit(X_train, y_train)
        elif HAS_XGBOOST:
            print("[Model] LightGBM unavailable. Initializing XGBoost Classifier backend...")
            self.backend = "xgboost"
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_seed,
                n_jobs=-1
            )
            if X_val is not None and y_val is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
        else:
            print("[Model] Initializing sklearn HistGradientBoostingClassifier backend...")
            self.backend = "hist_gb"
            self.model = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=6,
                max_leaf_nodes=31,
                random_state=self.random_seed
            )
            self.model.fit(X_train, y_train)
        print(f"[Model] Training successfully completed using backend: {self.backend}")
        return self
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
        probs = self.model.predict_proba(X)
        return probs[:, 1]
    def save(
        self,
        model_path: str = MODEL_PATH,
        metadata_path: str = MODEL_METADATA_PATH
    ) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        metadata = {
            "backend": self.backend,
            "feature_names": self.feature_names,
            "best_threshold": self.best_threshold,
            "random_seed": self.random_seed
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Model] Saved trained model to {model_path} and metadata to {metadata_path}")
    @classmethod
    def load(
        cls,
        model_path: str = MODEL_PATH,
        metadata_path: str = MODEL_METADATA_PATH
    ) -> "EntityResolutionModel":
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model or metadata file not found at {model_path}, {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        inst = cls(random_seed=metadata.get("random_seed", RANDOM_SEED))
        inst.backend = metadata.get("backend", "unknown")
        inst.feature_names = metadata.get("feature_names", [])
        inst.best_threshold = metadata.get("best_threshold", 0.50)
        with open(model_path, "rb") as f:
            inst.model = pickle.load(f)
        print(f"[Model] Successfully loaded model from {model_path} (Backend: {inst.backend})")
        return inst
