from typing import Any, Callable, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression


class LogisticRegressionSub(LogisticRegression):  # type: ignore[misc]
    # copy constructor with additional param transform_X
    def __init__(
        self,
        transform_X: Callable[[pd.DataFrame], pd.DataFrame],
        penalty: Any = "l2",
        dual: Any = False,
        tol: Any = 1e-4,
        C: Any = 1.0,
        fit_intercept: Any = True,
        intercept_scaling: Any = 1,
        class_weight: Any = None,
        random_state: Any = None,
        solver: Any = "lbfgs",
        max_iter: Any = 100,
        multi_class: Any = "auto",
        verbose: Any = 0,
        warm_start: Any = False,
        n_jobs: Any = None,
        l1_ratio: Any = None,
    ) -> None:
        super(LogisticRegressionSub, self).__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

        self.transform_X = transform_X

    def fit(
        self, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None
    ) -> "LogisticRegressionSub":
        super(LogisticRegressionSub, self).fit(self.transform_X(X), y, sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        return super(LogisticRegressionSub, self).predict(self.transform_X(X))

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return super(LogisticRegressionSub, self).predict_proba(self.transform_X(X))

    def predict_log_proba(self, X: pd.DataFrame) -> Any:
        return super(LogisticRegressionSub, self).predict_log_proba(self.transform_X(X))

    def score(
        self, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return super(LogisticRegressionSub, self).score(
            self.transform_X(X), y, sample_weight
        )

    def decision_function(self, X: pd.DataFrame) -> Any:
        return super(LogisticRegressionSub, self).decision_function(self.transform_X(X))


class DefenceLogReg(LogisticRegressionSub):
    model_name = "defence_logistic_regression"
    model_colums = model_colums = ["DREB", "STL", "BLK", "OPP_OREB", "PF"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        penalty: Any = "l2",
        dual: Any = False,
        tol: Any = 1e-4,
        C: Any = 1.0,
        fit_intercept: Any = True,
        intercept_scaling: Any = 1,
        class_weight: Any = None,
        random_state: Any = None,
        solver: Any = "lbfgs",
        max_iter: Any = 100,
        multi_class: Any = "auto",
        verbose: Any = 0,
        warm_start: Any = False,
        n_jobs: Any = None,
        l1_ratio: Any = None,
    ) -> None:
        super(DefenceLogReg, self).__init__(
            transform_X=DefenceLogReg.transform_X,
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )


class OffenceLogReg(LogisticRegressionSub):
    model_name = "offence_logistic_regression"
    model_colums = [
        "PTS",
        "FGM",
        "FG3M",
        "FTM",
        "OREB",
        "AST",
        "TOV",
        "OPP_DREB",
        "OPP_PF",
    ]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        penalty: Any = "l2",
        dual: Any = False,
        tol: Any = 1e-4,
        C: Any = 1.0,
        fit_intercept: Any = True,
        intercept_scaling: Any = 1,
        class_weight: Any = None,
        random_state: Any = None,
        solver: Any = "lbfgs",
        max_iter: Any = 100,
        multi_class: Any = "auto",
        verbose: Any = 0,
        warm_start: Any = False,
        n_jobs: Any = None,
        l1_ratio: Any = None,
    ) -> None:
        super(OffenceLogReg, self).__init__(
            transform_X=OffenceLogReg.transform_X,
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )


class EfficiencyLogReg(LogisticRegressionSub):
    model_name = "efficiency_logistic_regression"
    model_colums = ["FG_PCT", "FGA", "FG3_PCT", "FG3A", "FT_PCT", "FTA"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        penalty: Any = "l2",
        dual: Any = False,
        tol: Any = 1e-4,
        C: Any = 1.0,
        fit_intercept: Any = True,
        intercept_scaling: Any = 1,
        class_weight: Any = None,
        random_state: Any = None,
        solver: Any = "lbfgs",
        max_iter: Any = 100,
        multi_class: Any = "auto",
        verbose: Any = 0,
        warm_start: Any = False,
        n_jobs: Any = None,
        l1_ratio: Any = None,
    ) -> None:
        super(EfficiencyLogReg, self).__init__(
            transform_X=EfficiencyLogReg.transform_X,
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )


class GeneralLogReg(LogisticRegressionSub):
    model_name = "general_logistic_regression"
    model_colums = [
        "PTS",
        "OPP_PTS",
        "PF",
        "PLUS_MINUS",
        "OPP_PLUS_MINUS",
        "ALLOWED_PTS",
        "OPP_ALLOWED_PTS",
        "ALLOWED_PF",
        "ALLOWED_PLUS_MINUS",
        "OPP_ALLOWED_PLUS_MINUS",
    ]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        penalty: Any = "l2",
        dual: Any = False,
        tol: Any = 1e-4,
        C: Any = 1.0,
        fit_intercept: Any = True,
        intercept_scaling: Any = 1,
        class_weight: Any = None,
        random_state: Any = None,
        solver: Any = "lbfgs",
        max_iter: Any = 100,
        multi_class: Any = "auto",
        verbose: Any = 0,
        warm_start: Any = False,
        n_jobs: Any = None,
        l1_ratio: Any = None,
    ) -> None:
        super(GeneralLogReg, self).__init__(
            transform_X=GeneralLogReg.transform_X,
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )


class ShootingLogReg(LogisticRegressionSub):
    model_name = "shooting_logistic_regression"
    model_columns = [
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "ALLOWED_FGM",
        "ALLOWED_FGA",
        "ALLOWED_FG_PCT",
        "ALLOWED_FG3M",
        "ALLOWED_FG3A",
        "ALLOWED_FG3_PCT",
        "ALLOWED_FTM",
        "ALLOWED_FTA",
        "ALLOWED_FT_PCT",
        "OPP_FGM",
        "OPP_FGA",
        "OPP_FG_PCT",
        "OPP_FG3M",
        "OPP_FG3A",
        "OPP_FG3_PCT",
        "OPP_FTM",
        "OPP_FTA",
        "OPP_FT_PCT",
        "OPP_ALLOWED_FGM",
        "OPP_ALLOWED_FGA",
        "OPP_ALLOWED_FG_PCT",
        "OPP_ALLOWED_FG3M",
        "OPP_ALLOWED_FG3A",
        "OPP_ALLOWED_FG3_PCT",
        "OPP_ALLOWED_FTM",
        "OPP_ALLOWED_FTA",
        "OPP_ALLOWED_FT_PCT",
    ]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_columns]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        penalty: Any = "l2",
        dual: Any = False,
        tol: Any = 1e-4,
        C: Any = 1.0,
        fit_intercept: Any = True,
        intercept_scaling: Any = 1,
        class_weight: Any = None,
        random_state: Any = None,
        solver: Any = "lbfgs",
        max_iter: Any = 100,
        multi_class: Any = "auto",
        verbose: Any = 0,
        warm_start: Any = False,
        n_jobs: Any = None,
        l1_ratio: Any = None,
    ) -> None:
        super(ShootingLogReg, self).__init__(
            transform_X=ShootingLogReg.transform_X,
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )


# class
