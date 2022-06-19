from typing import List
import os

import pytest

import pandas as pd


@pytest.fixture(scope="module")
def categorical_preds():
    return ["Sex"]


@pytest.fixture(scope="module")
def numerical_preds():
    return ["Pclass", "Age", "SibSp", "Parch", "Fare"]


@pytest.fixture(scope="module")
def target_name():
    return "Survived"


@pytest.fixture(scope="module")
def input_df(
    categorical_preds: List[str], numerical_preds: List[str], target_name: str
):
    return pd.read_csv(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "sample_dataset.csv"),
        usecols=categorical_preds + numerical_preds + [target_name],
    )


@pytest.fixture(scope="module")
def encoded_input_df(input_df: pd.DataFrame):
    return input_df.replace({"female": 1, "male": 0})


@pytest.fixture(scope="module")
def imputed_input_df(encoded_input_df: pd.DataFrame):
    return encoded_input_df.fillna(0)
