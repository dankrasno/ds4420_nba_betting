from sklearn.neighbors import KNeighborsClassifier
from nba_betting.model.classes.base import Model


class DefenceKNeighbors(Model):
    model_name = "defence_logistic_kneighbors"
    model_columns = ["DREB", "STL", "BLK"]

    def __init__(self) -> None:
        super().__init__(
            DefenceKNeighbors.model_name,
            KNeighborsClassifier(),
            DefenceKNeighbors.model_columns,
        )


class OffenceKNeighbors(Model):
    model_name = "offence_logistic_kneighbors"
    model_columns = ["PTS", "FGM", "FG3M", "FTM", "OREB", "AST", "TOV"]

    def __init__(self) -> None:
        super().__init__(
            OffenceKNeighbors.model_name,
            KNeighborsClassifier(),
            OffenceKNeighbors.model_columns,
        )


class EfficiencyKNeighbors(Model):
    model_name = "efficiency_logistic_kneighbors"
    model_columns = ["FG_PCT", "FG3_PCT", "FT_PCT"]

    def __init__(self) -> None:
        super().__init__(
            EfficiencyKNeighbors.model_name,
            KNeighborsClassifier(),
            EfficiencyKNeighbors.model_columns,
        )

