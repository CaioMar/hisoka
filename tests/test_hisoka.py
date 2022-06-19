from itertools import product
from typing import List

import pytest


import pandas as pd

import hisoka.preprocessing as pp
from hisoka import __version__


rfs_parameters = ["model", "importance_method", "rand_var_type", "problem"]
model = ["catboost", "xgboost", "random_forest"]
importance_method = ["shap", "embedded"]
rand_var_type = ["float", "integer"]
problem = ["regression", "classification"]

rfs_args_list = []
for keys, values in product(
    [rfs_parameters], product(model, importance_method, rand_var_type, problem)
):
    arguments = dict(zip(keys, values))
    arguments["categorical_columns"] = ["Sex"]
    arguments["number_of_fits"] = 3
    rfs_args_list.append(arguments)

RANDOM_FEATURE_SELECTORS = [
    pp.RandomFeatureSelector(**arguments) for arguments in rfs_args_list
]

CATBOOST_RANDOM_FEATURE_SELECTORS = filter(
    lambda x: x.model == "catboost", RANDOM_FEATURE_SELECTORS
)
XGBOOST_RANDOM_FEATURE_SELECTORS = filter(
    lambda x: x.model == "xgboost", RANDOM_FEATURE_SELECTORS
)
RF_RANDOM_FEATURE_SELECTORS = filter(
    lambda x: x.model == "random_forest", RANDOM_FEATURE_SELECTORS
)


def test_version():
    assert __version__ == "0.1.0"


@pytest.mark.parametrize("RandomFeatureSelector", RF_RANDOM_FEATURE_SELECTORS)
def test_random_forest_rf_selector_fit_with_nans_should_fail(
    encoded_input_df: pd.DataFrame,
    RandomFeatureSelector: pp.RandomFeatureSelector,
    numerical_preds: List[str],
    categorical_preds: List[str],
    target_name: str,
) -> None:
    with pytest.raises(ValueError) as e:
        rfs = RandomFeatureSelector
        rfs.fit_transform(
            encoded_input_df[numerical_preds + categorical_preds],
            encoded_input_df[target_name],
        )
    assert "NaN" in str(e.value)


@pytest.mark.parametrize("RandomFeatureSelector", XGBOOST_RANDOM_FEATURE_SELECTORS)
def test_xgboost_rf_selector_fit_with_cat_col_should_fail(
    input_df: pd.DataFrame,
    RandomFeatureSelector: pp.RandomFeatureSelector,
    numerical_preds: List[str],
    categorical_preds: List[str],
    target_name: str,
) -> None:
    with pytest.raises(ValueError) as e:
        rfs = RandomFeatureSelector
        rfs.fit_transform(
            input_df[numerical_preds + categorical_preds], input_df[target_name]
        )
    assert categorical_preds[0] in str(e.value)


@pytest.mark.parametrize("RandomFeatureSelector", RANDOM_FEATURE_SELECTORS)
def test_rf_selector_fit_transform_returns_dataframe(
    imputed_input_df: pd.DataFrame,
    RandomFeatureSelector: pp.RandomFeatureSelector,
    numerical_preds: List[str],
    categorical_preds: List[str],
    target_name: str,
) -> None:
    rfs = RandomFeatureSelector
    output_df = rfs.fit_transform(
        imputed_input_df[numerical_preds + categorical_preds],
        imputed_input_df[target_name],
    )
    assert isinstance(output_df, pd.DataFrame)
