from sklearn.linear_model import LogisticRegression

from nba_betting.model.classes.base import Model


class DefenceLogReg(Model):
    model_name = "defence_logistic_regression"
    model_colums = ["DREB", "STL", "BLK"]

    def __init__(self) -> None:
        super().__init__(
            DefenceLogReg.model_name,
            LogisticRegression(),
            DefenceLogReg.model_colums,
        )


class OffenceLogReg(Model):
    model_name = "offence_logistic_regression"
    model_colums = ["PTS", "FGM", "FG3M", "FTM", "OREB", "AST", "TOV"]

    def __init__(self) -> None:
        super().__init__(
            OffenceLogReg.model_name,
            LogisticRegression(),
            OffenceLogReg.model_colums,
        )


class EfficiencyLogReg(Model):
    model_name = "efficiency_logistic_regression"
    model_colums = ["FG_PCT", "FG3_PCT", "FT_PCT"]

    def __init__(self) -> None:
        super().__init__(
            EfficiencyLogReg.model_name,
            LogisticRegression(),
            EfficiencyLogReg.model_colums,
        )


# TODO: encode the team and home/travel data to
# create another simple regressor
