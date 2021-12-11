
import pandas as pd
import numpy as np
from config import *

def get_data(dir, file_name):
    """
    :param dir: directory from which file is loaded
    :param file_name: file to load
    :return: dataset for drift detection, without irrelevant column, and
    with time column formatted as such.
    """
    df = pd.read_csv(f'{dir}/{file_name}?raw=true')
    df['created_at'] = pd.to_datetime(df[TIME_COL])

    df = df.drop('order_id', axis=1)

    return df

def ohe(data, col_to_ohe):
    """
    replace categorical columns with one hot encoded dataframe
    :param data: dataframe with categorical columns
    :param col_to_ohe: what column to transform
    :return: dataframe with ohe instead of categorical data
    """
    return pd.get_dummies(data, columns=col_to_ohe)


def get_batch(data):
    """
    Divides data into batches of samples within same time frame (TIME_COL).
    :param data: dataframe with time column.
    :return: input dataframe, with batch column added.
    """
    data.sort_values(TIME_COL, inplace=False)
    data['batch'] = data.groupby(TIME_COL).cumcount() // BATCH_SIZE

    return data


def to_batch(cat_data, num_data):
    """
    Aggregates data into one sample of time X batch.
    For categorical data, values are counted
    For numerical data, values are averaged
    :param cat_data: dataframe of categorical features. must include time and batch columns
    :param num_data: dataframe of numerical features. must include time and batch columns
    :return: one dataframe where each line is the aggregate measure of its time X batch
    """

    batched_num = num_data.groupby([TIME_COL, 'batch'])\
        .mean().reset_index()
    batched_cat = cat_data.groupby([TIME_COL, 'batch'])\
        .sum().reset_index()

    batched = batched_num.merge(
        batched_cat,
        left_on=['batch', 'created_at'],
        right_on=['batch', 'created_at']
    )

    batched = batched.drop('batch', axis=1)

    return batched


def split_cat_num(data, CAT_FEATURES):
    """
    :param data: dataframe of numerical and categorical features.
    :param CAT_FEATURES: categorical features to be split from numerical.
    names must be included in the columns of initial dataframe
    :return: 2 dataframes: categorical, and numeric
    """

    cat_data = data[[TIME_COL, 'batch']+CAT_FEATURES]
    num_data = data.drop(CAT_FEATURES, axis=1)

    return cat_data, num_data


def split_date_data(data, time_col):
    """
    :param data: dataframe of features to test for drift
    :param time_col: name of column in dataframe 'data' indexing time
    :return: 2 dataframes of features and their corresponding stream time
    """

    dates = data[time_col]
    data = data.drop(time_col, axis=1)

    return dates, data


def editting(data):
    """
    function to pipe some of our function in order to transform the data
    :param data: raw data
    :return: processed dataframe to feed to detector
    """
    data = data.drop('score', axis=1)
    with_batches = get_batch(data)
    cat_data, num_data = split_cat_num(with_batches, CAT_FEATURES)
    oh_cat = ohe(cat_data, CAT_FEATURES)
    batched = to_batch(oh_cat, num_data)
    dates, data = split_date_data(batched, TIME_COL)
    return dates, data
