import argparse
import os
import sys

sys.path.append(os.getcwd())  # sys path needed to import other classes
from nba_betting.api.data import year_to_reg_season_start
from nba_betting.model.training import fit_all


class ModelTrainer:
    def __init__(self) -> None:
        args = self.parse_args()
        self.train_year: int = args.train_year
        self.test_year: int = args.test_year
        self.no_store_cache: bool = args.no_store_cache
        self.no_load_cache: bool = args.no_load_cache

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--train-year",
            required=True,
            type=int,
            choices=[_ for _ in year_to_reg_season_start.keys()],
        )
        parser.add_argument(
            "--test-year",
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
            train_year=self.train_year,
            test_year=self.test_year,
            load_cache=not self.no_load_cache,
            store_cache=not self.no_store_cache,
        )


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.run()
