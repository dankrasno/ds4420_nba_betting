import argparse
from typing import List

import sys
import os
sys.path.append(os.getcwd()) # sys path needed to import other classes

from nba_betting.api.data import year_to_reg_season_start
from nba_betting.model.classes import NBA_MODELS, NBAModel
from nba_betting.model.training import fit_all


class ModelTrainer:
    def __init__(self) -> None:
        args = self.parse_args()
        self.models: List[NBAModel] = [NBA_MODELS[model]() for model in args.models]
        self.data_year: int = args.data_year
        self.no_store_cache: bool = args.no_store_cache
        self.no_load_cache: bool = args.no_load_cache

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--models",
            nargs=argparse.ONE_OR_MORE,
            type=str,
            choices=[_ for _ in NBA_MODELS.keys()],
        )
        parser.add_argument(
            "--data-year",
            required=True,
            type=int,
            choices=[_ for _ in year_to_reg_season_start.keys()],
        )
        parser.add_argument(
            "--no-store-cache",
            action=argparse._StoreTrueAction,
            help="Do not store trained models to cache",
        )
        parser.add_argument(
            "--no-load-cache",
            action=argparse._StoreTrueAction,
            help="Do not load trained models from cache",
        )
        return parser.parse_args()

    def run(self) -> None:
        fit_all(
            models=self.models,
            year=self.data_year,
            load_cache=not self.no_load_cache,
            store_cache=not self.no_store_cache,
        )


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.run()
