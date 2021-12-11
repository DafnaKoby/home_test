import numpy as np
import pandas as pd
from config import *


def ohe(data, added_prefix, col_to_ohe):
    """
    replace categorical columns with one hot encoded dataframe
    :param data: dataframe with categorical columns
    :param added_prefix: what prefix to add to ohe columns
    :param col_to_ohe: what column to transform
    :return: dataframe with ohe instead of categorical data
    """
    return pd.get_dummies(data, prefix=added_prefix, columns=col_to_ohe, drop_first=True)


def editing(data):
    """
    function to pipe some of our function in order to transform the data
    :return: dataframe
    """
    df_transformed = ohe(data, CAT_FEATURES)
    return df_transformed

def get_unique_values(data, cols):
    cols = data[cols].unique()
    return cols