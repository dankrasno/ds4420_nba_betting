import pickle
from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

CACHE_DIRECTORY = Path("data/model_cache")
EstimatorType = Union[LinearRegression, LogisticRegression]


class Model:
    def __init__(
        self,
        name: str,
        estimator: EstimatorType,
        column_filter: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.estimator = estimator
        self.column_filter = column_filter
        self.model_cache_dir = CACHE_DIRECTORY / self.name

    def store_estimator_cache(self, training_set_id: str) -> None:
        """Saves the estimator to cache

        Args:
            training_set_id (str): An ID representing the training set
                the estimator was trained on
        """
        training_cache_dir = self.model_cache_dir / training_set_id

        # save the estimator binary to disk
        model_filepath = training_cache_dir / "estimator.sav"
        model_filepath.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.estimator, open(model_filepath, "wb"))

    def load_estimator_cache(self, training_set_id: str) -> bool:
        """Loads the estimator from cache

        Args:
            training_set_id (str): An ID representing the training set
                the cached estimator was trained on
        """
        training_cache_dir = self.model_cache_dir / training_set_id

        # load the extimator binary from disk
        model_filepath = training_cache_dir / "estimator.sav"
        cache_found = model_filepath.resolve().is_file()
        if cache_found:
            self.estimator = pickle.load(open(model_filepath, "rb"))

        return cache_found

    def fit(self, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None) -> Any:
        return self.estimator.fit(
            X=X[self.column_filter] if self.column_filter else X,
            y=y,
            sample_weight=sample_weight,
        )

    def predict(self, X: pd.DataFrame) -> Any:
        return self.estimator.predict(
            X=X[self.column_filter] if self.column_filter else X
        )

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return self.estimator.predict_proba(
            X=X[self.column_filter] if self.column_filter else X
        )

    def predict_log_proba(self, X: pd.DataFrame) -> Any:
        return self.estimator.predict_log_proba(
            X=X[self.column_filter] if self.column_filter else X
        )
