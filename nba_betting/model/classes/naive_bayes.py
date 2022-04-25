from typing import Any, Callable, Optional
import pandas as pd
from sklearn.naive_bayes import GaussianNB


class GaussianNBSub(GaussianNB):  # type: ignore[misc]
    # copy constructor with additional param transform_X
    def __init__(
        self,
        transform_X: Callable[[pd.DataFrame], pd.DataFrame],
        priors: Any = None, 
        var_smoothing: Any = 1e-9,

    ) -> None:
        super(GaussianNBSub, self).__init__(
            priors = priors,
            var_smoothing = var_smoothing
        )

        self.transform_X = transform_X

    def fit(
        self, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None
    ) -> "GaussianNBSub":
        super(GaussianNBSub, self).fit(self.transform_X(X), y, sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        return super(GaussianNBSub, self).predict(self.transform_X(X))

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return super(GaussianNBSub, self).predict_proba(self.transform_X(X))

    def predict_log_proba(self, X: pd.DataFrame) -> Any:
        return super(GaussianNBSub, self).predict_log_proba(self.transform_X(X))

    def score(
        self, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return super(GaussianNBSub, self).score(
            self.transform_X(X), y, sample_weight
        )

    def decision_function(self, X: pd.DataFrame) -> Any:
        return super(GaussianNBSub, self).decision_function(self.transform_X(X))


class DefenceGaussianNB(GaussianNBSub):
    model_name = "defence_gaussian_naive_bayes"
    model_colums = ["DREB", "STL", "BLK"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        priors: Any = None, 
        var_smoothing: Any = 1e-9,
    ) -> None:
        super(DefenceGaussianNB, self).__init__(
            transform_X=DefenceGaussianNB.transform_X,
            priors=priors,
            var_smoothing=var_smoothing,
        )


class OffenceGaussianNB(GaussianNBSub):
    model_name = "offence_gaussian_naive_bayes"
    model_colums = ["PTS", "FGM", "FG3M", "FTM", "OREB", "AST", "TOV"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        priors: Any = None, 
        var_smoothing: Any = 1e-9,
    ) -> None:
        super(OffenceGaussianNB, self).__init__(
            transform_X=OffenceGaussianNB.transform_X,
            priors=priors,
            var_smoothing=var_smoothing,
        )


class EfficiencyGaussianNB(GaussianNBSub):
    model_name = "efficiency_gaussian_naive_bayes"
    model_colums = ["FG_PCT", "FGA", "FG3_PCT", "FG3A", "FT_PCT", "FTA"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        priors: Any = None, 
        var_smoothing: Any = 1e-9,
    ) -> None:
        super(EfficiencyGaussianNB, self).__init__(
            transform_X=EfficiencyGaussianNB.transform_X,
            priors=priors,
            var_smoothing=var_smoothing,
        )


class GeneralGaussianNB(GaussianNBSub):
    model_name = "general_gaussian_naive_bayes"
    model_colums = model_colums = ["PTS", "OPP_PTS", "PF", "PLUS_MINUS", "OPP_PLUS_MINUS"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        priors: Any = None, 
        var_smoothing: Any = 1e-9,
    ) -> None:
        super(GeneralGaussianNB, self).__init__(
            transform_X=GeneralGaussianNB.transform_X,
            priors=priors,
            var_smoothing=var_smoothing,
        )
