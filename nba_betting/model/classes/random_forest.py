from typing import Any, Callable, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class RandomForestSub(RandomForestClassifier):  # type: ignore[misc]
    # copy constructor with additional param transform_X
    def __init__(
        self,
        transform_X: Callable[[pd.DataFrame], pd.DataFrame],
        n_estimators: Any =100,
        criterion: Any ="gini",
        max_depth: Any =None,
        min_samples_split: Any =2,
        min_samples_leaf: Any =1,
        min_weight_fraction_leaf: Any =0.0,
        max_features: Any ="auto",
        max_leaf_nodes: Any =None,
        min_impurity_decrease: Any =0.0,
        bootstrap: Any =True,
        oob_score: Any =False,
        n_jobs: Any =None,
        random_state: Any =None,
        verbose: Any =0,
        warm_start: Any =False,
        class_weight: Any =None,
        ccp_alpha: Any =0.0,
        max_samples: Any =None,
    ) -> None:
        super(RandomForestSub, self).__init__(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose,
        warm_start=warm_start,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples,
        )

        self.transform_X = transform_X

    def fit(
        self, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None
    ) -> "RandomForestSub":
        super(RandomForestSub, self).fit(self.transform_X(X), y, sample_weight)
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        return super(RandomForestSub, self).predict(self.transform_X(X))

    def predict_proba(self, X: pd.DataFrame) -> Any:
        return super(RandomForestSub, self).predict_proba(self.transform_X(X))

    def predict_log_proba(self, X: pd.DataFrame) -> Any:
        return super(RandomForestSub, self).predict_log_proba(self.transform_X(X))

    def score(
        self, X: pd.DataFrame, y: Any, sample_weight: Optional[Any] = None
    ) -> Any:
        return super(RandomForestSub, self).score(
            self.transform_X(X), y, sample_weight
        )

    def decision_function(self, X: pd.DataFrame) -> Any:
        return super(RandomForestSub, self).decision_function(self.transform_X(X))


class DefenceRandForest(RandomForestSub):
    model_name = "defence_random_forest"
    model_colums = ["DREB", "STL", "BLK"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        n_estimators: Any =100,
        criterion: Any ="gini",
        max_depth: Any =None,
        min_samples_split: Any =2,
        min_samples_leaf: Any =1,
        min_weight_fraction_leaf: Any =0.0,
        max_features: Any ="auto",
        max_leaf_nodes: Any =None,
        min_impurity_decrease: Any =0.0,
        bootstrap: Any =True,
        oob_score: Any =False,
        n_jobs: Any =None,
        random_state: Any =None,
        verbose: Any =0,
        warm_start: Any =False,
        class_weight: Any =None,
        ccp_alpha: Any =0.0,
        max_samples: Any =None,
    ) -> None:
        super(DefenceRandForest, self).__init__(
            transform_X=DefenceRandForest.transform_X,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )


class OffenceRandForest(RandomForestSub):
    model_name = "offence_random_forest"
    model_colums = ["PTS", "FGM", "FG3M", "FTM", "OREB", "AST", "TOV"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        n_estimators: Any =100,
        criterion: Any ="gini",
        max_depth: Any =None,
        min_samples_split: Any =2,
        min_samples_leaf: Any =1,
        min_weight_fraction_leaf: Any =0.0,
        max_features: Any ="auto",
        max_leaf_nodes: Any =None,
        min_impurity_decrease: Any =0.0,
        bootstrap: Any =True,
        oob_score: Any =False,
        n_jobs: Any =None,
        random_state: Any =None,
        verbose: Any =0,
        warm_start: Any =False,
        class_weight: Any =None,
        ccp_alpha: Any =0.0,
        max_samples: Any =None,
    ) -> None:
        super(OffenceRandForest, self).__init__(
            transform_X=OffenceRandForest.transform_X,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )


class EfficiencyRandForest(RandomForestSub):
    model_name = "efficiency_random_forest"
    model_colums = ["FG_PCT", "FGA", "FG3_PCT", "FG3A", "FT_PCT", "FTA"]

    @classmethod
    def transform_X(cls, X: pd.DataFrame) -> pd.DataFrame:
        return X[cls.model_colums]

    # copy constructor to make sklearn happy
    def __init__(
        self,
        n_estimators: Any =100,
        criterion: Any ="gini",
        max_depth: Any =None,
        min_samples_split: Any =2,
        min_samples_leaf: Any =1,
        min_weight_fraction_leaf: Any =0.0,
        max_features: Any ="auto",
        max_leaf_nodes: Any =None,
        min_impurity_decrease: Any =0.0,
        bootstrap: Any =True,
        oob_score: Any =False,
        n_jobs: Any =None,
        random_state: Any =None,
        verbose: Any =0,
        warm_start: Any =False,
        class_weight: Any =None,
        ccp_alpha: Any =0.0,
        max_samples: Any =None,
    ) -> None:
        super(EfficiencyRandForest, self).__init__(
            transform_X=EfficiencyRandForest.transform_X,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )


# TODO: encode the team and home/travel data to
# create another simple regressor
