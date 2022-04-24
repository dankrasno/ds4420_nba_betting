import argparse

from nba_betting.api.data import year_to_reg_season_start
from nba_betting.model.training import fit_ensemble


class ModelTrainer:
    def __init__(self) -> None:
        args = self.parse_args()
        self.train_year: int = args.train_year
        self.test_year: int = args.test_year

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
        return parser.parse_args()

    def run(self) -> None:
        fit_ensemble(
            train_year=self.train_year,
            test_year=self.test_year,
        )


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.run()
