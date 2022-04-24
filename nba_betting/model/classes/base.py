import pickle
from pathlib import Path
from typing import Optional, Union, cast

from sklearn.linear_model import LinearRegression, LogisticRegression

from nba_betting.model.classes import NBAModel

CACHE_DIRECTORY = Path("data/model_cache")
EstimatorType = Union[LinearRegression, LogisticRegression]


def store_estimator_cache(model: NBAModel, training_set_id: str) -> None:
    """Saves the estimator to cache

    Args:
        training_set_id (str): An ID representing the training set
            the estimator was trained on
    """
    training_cache_dir = CACHE_DIRECTORY / model.model_name / training_set_id

    # save the estimator binary to disk
    model_filepath = training_cache_dir / "estimator.sav"
    model_filepath.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(model_filepath, "wb"))


def load_estimator_cache(model_name: str, training_set_id: str) -> Optional[NBAModel]:
    """Loads the estimator from cache

    Args:
        training_set_id (str): An ID representing the training set
            the cached estimator was trained on
    """
    training_cache_dir = CACHE_DIRECTORY / model_name / training_set_id

    # load the extimator binary from disk
    model_filepath = training_cache_dir / "estimator.sav"
    cache_found = model_filepath.resolve().is_file()
    if cache_found:
        return cast(NBAModel, pickle.load(open(model_filepath, "rb")))

    return None
