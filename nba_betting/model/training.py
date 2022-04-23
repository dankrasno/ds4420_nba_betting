from typing import List

from nba_betting.api.data import get_games_by_year, make_xy
from nba_betting.logging.tools import logger
from nba_betting.model.classes.base import Model


def fit_all(
    models: List[Model],
    year: int,
    load_cache: bool = True,
    store_cache: bool = True,
) -> None:
    logger.info("Using data for year %s as training set", year)
    g = get_games_by_year(year)
    X, y = make_xy(g)
    logger.info(
        "Retrieved data for year %s.\nColumns: %r\nTarget: %r",
        year,
        X.columns,
        y.head(),
    )

    logger.debug(X)
    logger.debug(y)

    training_set_id = f"train-set-{year}"
    logger.debug("Training set ID = {training_set_id}")
    logger.info("==================================================")
    logger.info("Training models: %r", [model.name for model in models])

    for model in models:
        if load_cache:
            loaded = model.load_estimator_cache(training_set_id)
            if loaded:
                logger.info(
                    "Loaded trained model %s with estimator %r",
                    model.name,
                    model.estimator,
                )
                continue

        logger.info("Training %s with estimator %r", model.name, model.estimator)
        model.fit(X, y)
        logger.info("Completed training %s", model.name)
        if store_cache:
            logger.info("Saving trained model %s to cache", model.name)
            model.store_estimator_cache(training_set_id)
