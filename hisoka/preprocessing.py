from typing import Optional, List, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from xgboost import XGBClassifier, XGBRegressor

import shap


class RandomFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects features that have greater importance than random features.

    This selector fits estimators on the dataset with a random feature
    added to the mix. The subset of features that consistently showcase
    higher importance than the random features are kept. The other subset
    is droped.

    Parameters
    ----------
    rand_var_type: 'integer', 'float', default='integer'
        Type of random feature being generated. Either float
        or integer type.
    low_end: int, default=0
        Lower cap of the random feature.
    high_end: int, default=1
        Higher cap of the random feature.
    model: 'random_forest', 'catboost', 'xgboost', default='random_forest'
        Estimator used to compute feature importances.
    problem: 'classification', 'regression', default='classification'
        Type of supervised learning problem.
    importance_method: 'shap', 'embedded', default='shap'
        Represent the importance method used to compute feature
        importances. 'embedded' option use the feature importance
        of the algorithms instead of using Shapley Values.
    number_of_fits: int, default=1
    categorical_columns: List of columns that are categorical.
    random_state: int, default=42
    """

    def __init__(
        self,
        rand_var_type: str = "integer",
        low_end: int = 0,
        high_end: int = 10,
        model: str = "random_forest",
        problem: str = "classification",
        importance_method: str = "shap",
        number_of_fits: int = 1,
        categorical_columns: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        self.rand_var_type = rand_var_type
        self.low_end = low_end
        self.high_end = high_end
        self.model = model
        self.problem = problem
        self.importance_method = importance_method
        self.number_of_fits = number_of_fits
        self.categorical_columns = categorical_columns
        self.random_state = random_state

    def _get_model(
        self,
    ) -> Any:
        if self.problem == "classification":
            if self.model == "random_forest":
                return RandomForestClassifier(
                    class_weight="balanced",
                    n_estimators=50,
                    max_depth=5,
                    random_state=self.random_state,
                )
            elif self.model == "catboost":
                return CatBoostClassifier(
                    max_depth=5,
                    n_estimators=100,
                    rsm=0.8,
                    learning_rate=0.1,
                    cat_features=self.categorical_columns,
                    random_seed=self.random_state,
                    verbose=False,
                )
            elif self.model == "xgboost":
                return XGBClassifier(
                    objective="binary:logistic",
                    max_depth=5,
                    n_estimators=100,
                    colsample_bytree=0.8,
                )
        elif self.problem == "regression":
            if self.model == "random_forest":
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                )
            elif self.model == "catboost":
                return CatBoostRegressor(
                    max_depth=5,
                    n_estimators=100,
                    rsm=0.8,
                    learning_rate=0.1,
                    cat_features=self.categorical_columns,
                    random_seed=self.random_state,
                    verbose=False,
                )
            elif self.model == "xgboost":
                return XGBRegressor(
                    max_depth=5,
                    n_estimators=100,
                    colsample_bytree=0.8,
                )
            raise ValueError("Problem must be 'classification' or 'regression'.")
        raise ValueError("Model must be 'random_forest', 'catboost' or 'xgboost'.")

    def _random_feature(self, X: pd.DataFrame) -> None:
        if self.rand_var_type == "integer":
            X["rand_var"] = np.random.randint(
                self.low_end, self.high_end + 1, size=X.shape[0]
            )
        elif self.rand_var_type == "float":
            X["rand_var"] = (self.high_end + self.low_end) * np.random.random(
                size=X.shape[0]
            ) - self.low_end
        else:
            raise ValueError("Invalid rand_var_type. Must be 'integer' or 'float'.")

    def _random_feature_selector(
        self, X: pd.DataFrame, y: pd.Series, model: Any
    ) -> List[str]:
        if self.importance_method == "shap":
            shap_vals = None
            if not isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
                shap_vals = shap.TreeExplainer(model).shap_values(X, y)
            else:
                shap_vals = shap.TreeExplainer(model).shap_values(
                    Pool(X, y, cat_features=self.categorical_columns)
                )
            if isinstance(model, RandomForestClassifier):
                shap_vals = shap_vals[1]
            importance_matrix = np.dstack((X.columns, np.abs(shap_vals).mean(axis=0)))[
                0
            ]
        elif self.importance_method == "embedded":
            importance_matrix = np.dstack((X.columns, model.feature_importances_))[0]
        feature_importances = pd.DataFrame(
            importance_matrix,
            columns=["feature", "importance"],
        ).set_index("feature")
        rand_importance = feature_importances.loc["rand_var"][0]
        return list(feature_importances.query("importance > @rand_importance").index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomFeatureSelector":
        """Fits feature selector to the dataset storing the features that must be kept."""
        self.features_selected = X.columns
        for _ in tqdm(range(self.number_of_fits)):
            self._random_feature(X)
            model = self._get_model().fit(X, y)
            self.features_selected = list(
                set(self.features_selected).intersection(
                    set(self._random_feature_selector(X, y, model))
                )
            )
            X.drop("rand_var", axis=1, inplace=True)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Transform the dataset dropping all low importance features."""
        X = X[self.features_selected]
        return X
