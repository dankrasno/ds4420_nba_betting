# from itertools import combinations_with_replacement
from typing import Any, Dict, List, Tuple

# import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

from nba_betting.api.data import get_games_by_year, make_xy
from nba_betting.logging.tools import logger
from nba_betting.model.classes import NBA_MODELS
from nba_betting.model.classes.base import (
    load_estimator_cache,
    store_estimator_cache,
)
from nba_betting.model.classes.linear import (
    DefenceLogReg,
    EfficiencyLogReg,
    OffenceLogReg,
)


def preprocess_games(year: int) -> "Tuple[pd.DataFrame, pd.Series[str]]":
    logger.info("Processing data for year %s", year)

    games_df = get_games_by_year(year).reset_index()
    X, y = make_xy(games_df)

    logger.info(
        "Retrieved data for year %s.\nColumns: %r\nTarget: %r",
        year,
        X.columns,
        y.head(),
    )

    return X, y


def get_training_set_id(year: int) -> str:
    training_set_id = f"train-set-{year}"
    logger.debug(f"Training set ID = {training_set_id}")
    return training_set_id


def fit_all(
    train_year: int,
    test_year: int,
    load_cache: bool = True,
    store_cache: bool = True,
) -> None:
    models = [model() for model in NBA_MODELS.values()]
    X, y = preprocess_games(train_year)
    X_test, y_test = preprocess_games(test_year)
    training_set_id = get_training_set_id(train_year)

    logger.info("==================================================")

    logger.info("Training models: %r", [model.model_name for model in models])
    for model in models:
        loaded_cache = False
        if load_cache:
            loaded_model = load_estimator_cache(model.model_name, training_set_id)
            if loaded_model is not None:
                model = loaded_model
                loaded_cache = True
                logger.info(
                    "Loaded trained model %s with estimator %r",
                    model.model_name,
                    model,
                )
            else:
                logger.error("Failed to load model %s", model.model_name)
        if not loaded_cache:
            logger.info("Training %s", model.model_name)
            model = model.fit(X, y)
            logger.info("Completed training %s", model.model_name)

        logger.info("Training %s complete", model.model_name)
        logger.info("Training set score: %s", model.score(X, y))
        logger.info("Testing set score: %s", model.score(X_test, y_test))

        if store_cache:
            logger.info("Saving trained model %s to cache", model.model_name)
            store_estimator_cache(model, training_set_id)
        logger.info("")


def fit_ensemble(
    train_year: int,
    test_year: int,
) -> None:
    models = [model() for model in NBA_MODELS.values()]
    X, y = preprocess_games(train_year)
    X_test, y_test = preprocess_games(test_year)

    logger.info("==================================================")

    logger.info("Training ensemble with: %r", [model.model_name for model in models])

    voting_cls = VotingClassifier(
        estimators=[(model.model_name, model) for model in models],
        voting="soft",
    )

    param_grid: Dict[str, List[Any]] = {
        "weights": list(
            set(
                tuple(np.divide(weight_combination, sum(weight_combination)))
                for weight_combination in combinations_with_replacement(
                    [i for i in range(1,4)], len(models)
                )
                if sum(weight_combination) != 0
            )
        )
        # "lr__C": [1.0, 100.0], "rf__n_estimators": [20, 200]
    }
    logger.info("Using grid search wih params: %r", param_grid)
    grid_search: GridSearchCV = GridSearchCV(
        estimator=voting_cls, param_grid=param_grid, cv=5
    )
    grid_search = grid_search.fit(X, y)

    logger.info(
        "Grid search complete. GridSearchCV best params: %r", grid_search.best_params_
    )

    logger.info("Training set score: %s", grid_search.score(X, y))
    logger.info("Testing set score: %s", grid_search.score(X_test, y_test))
