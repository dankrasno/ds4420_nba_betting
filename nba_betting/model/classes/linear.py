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
    model_colums = ["DREB", "STL", "BLK"]

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
    model_colums = ["PTS", "FGM", "FG3M", "FTM", "OREB", "AST", "TOV"]

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
    model_colums = ["FG_PCT", "FG3_PCT", "FT_PCT"]

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


# TODO: encode the team and home/travel data to
# create another simple regressor
