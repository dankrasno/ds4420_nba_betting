from typing import Any, List, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from nba_betting.model.classes import NBAModel

LearnerType = Union[Ridge, RandomForestRegressor]


class UnsupervisedEnsemble:
    def __init__(
        self,
        models: List[NBAModel],
        correction_learner: LearnerType,
        weight_learner: LearnerType,
    ) -> None:
        # sub classifier models
        self.models = {model.model_name: model for model in models}

        # model used to predict corrections for sub_models on X
        self.correction_model = correction_learner

        # inverse distance scaler used for transforming inverse
        # distance metric to the range [0, 1] for weighting
        self.inverse_distance_scaler = MinMaxScaler()
        # model used to predict optimal weights on X
        self.weight_learner = weight_learner

    def _fit_models(self, X: pd.DataFrame, y: Any) -> None:
        for model_name, model in self.models.items():
            self.models[model_name] = model.fit(X, y)

    def _fit_correction(self, X: pd.DataFrame, y: Any) -> None:
        # make predictions using the sub models, and
        # use predictions to calculate -1/1 for corrections
        corrections = pd.DataFrame()
        for model_name, model in self.models.items():
            corrections[model_name] = (y == model.predict(X)).astype(int) * 2 - 1

        # fit the correction model with the corrections
        self.correction_model = self.correction_model.fit(X, corrections)

    def _predict_correction(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            self.correction_model.predict(X),
            columns=[model_name for model_name in self.models],
        )

    def _predict_corrected_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        # predict the corrections on X using the correction model
        corrections = self._predict_correction(X)
        model_predictions = pd.DataFrame()
        for model_name, model in self.models.items():
            # predict W probablilty for each sub model, scale to [-1, 1], and
            # then multiply by corrections
            model_proba = (
                pd.DataFrame(model.predict_proba(X), columns=["L", "W"]) * 2 - 1
            )
            # multiply by the correction to get an adjusted proba in [-1, 1]
            corrected_proba = (
                corrections[model_name] * model_proba.W  # type:ignore[type-var]
            )
            # snap back to [0, 1] and set
            model_predictions[model_name] = (corrected_proba + 1) / 2

        return model_predictions

    def _get_inverse_distance(
        self, corrected_predictions: pd.DataFrame, y: Any
    ) -> pd.DataFrame:
        y_proba = pd.DataFrame()
        y_proba["L"] = (y == "L").astype(int)
        y_proba["W"] = (y == "W").astype(int)

        inv_distance = pd.DataFrame()
        for model_name in self.models.keys():
            model_distances = np.square(
                corrected_predictions[model_name] - y_proba.W  # type:ignore[type-var]
            )
            inv_distance[model_name] = np.nan_to_num(
                (1 / model_distances), nan=100, posinf=100, neginf=100
            )

        return inv_distance

    def _fit_weights(self, X: pd.DataFrame, y: Any) -> None:
        # get corrected predictions for "W" snapped to the range [0, 1]
        corrected_proba = self._predict_corrected_proba(X)
        # calculate the inverse distance between the true label and predictions
        inv_distance = self._get_inverse_distance(corrected_proba, y)

        # adjust the inverse distance to the range [0, 1]
        self.inverse_distance_scaler = MinMaxScaler()
        self.inverse_distance_scaler.fit(inv_distance)
        inv_distance.iloc[:, :] = self.inverse_distance_scaler.transform(inv_distance)

        # fit the weight learner to the adjusted inverse distance
        self.weight_learner.fit(X, inv_distance)

    def _predict_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        weights = pd.DataFrame(
            self.weight_learner.predict(X),
            columns=[model_name for model_name in self.models],
        )
        # ensure each row sums to 1 and return
        return weights.div(  # type: ignore [no-any-return]
            weights.sum(axis=1), axis=0  # type: ignore [operator]
        )

    def fit(self, X: pd.DataFrame, y: Any) -> None:
        # fit all sub models
        self._fit_models(X, y)
        # use the fitted sub models to fit the correction model
        self._fit_correction(X, y)
        # fit a weight model using the corrected predictions and X
        self._fit_weights(X, y)

    def predict(self, X: pd.DataFrame) -> Any:
        # get corrected predictions for "W" snapped to the range [0, 1]
        corrected_proba = self._predict_corrected_proba(X)
        # get the weight predictions on each row of X
        weights = self._predict_weights(X)
        # multiply and return the rowsums
        return (corrected_proba * weights).sum(axis=1)  # type: ignore [operator]
