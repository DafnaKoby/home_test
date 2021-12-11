
import pandas as pd
import numpy as np
from config import *

def get_data(dir, file_name):
    df = pd.read_csv(f'{dir}/{file_name}?raw=true')
    df['created_at'] = pd.to_datetime(df['created_at'])

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

    data.sort_values(TIME_COL, inplace=False)
    data['batch'] = data.groupby(TIME_COL).cumcount() // BATCH_SIZE

    return data


def to_batch(cat_data, num_data):

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

    cat_data = data[[TIME_COL, 'batch']+CAT_FEATURES]
    num_data = data.drop(CAT_FEATURES, axis=1)

    return cat_data, num_data


def split_date_data(data, time_col):

    dates = data[time_col]
    data = data.drop(time_col, axis=1)

    return dates, data


def editting(data):
    """
    function to pipe some of our function in order to transform the data
    :return: dataframe
    """
    data = data.drop('score', axis=1)
    with_batches = get_batch(data)
    cat_data, num_data = split_cat_num(with_batches, CAT_FEATURES)
    oh_cat = ohe(cat_data, CAT_FEATURES)
    batched = to_batch(oh_cat, num_data)
    dates, data = split_date_data(batched, TIME_COL)
    return dates, data
