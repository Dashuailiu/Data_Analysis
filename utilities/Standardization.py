import pandas
import operator
import numpy as np


def z_score(data, filter_col=None):
    """
    :param data
    :return: standardized
    """
    if not isinstance(data, pandas.DataFrame):
        raise TypeError("The type of input is wrong. Please use pandas.DataFrame.")

    if not filter_col:
        filter_col = []

    summary = data.describe()
    for i in range(len(data.columns)-1):
        if data.iloc[:, i].name == filter_col:
            continue
        mean = summary.iloc[1, i]
        std = summary.iloc[2, i]
        data.iloc[:, i] = (data.iloc[:, i] - mean) / std

    return data
    # return (data - data.mean())/data.std()


def z_score_np(data, ddof=1):
    """
    :param data:
    :param ddof: biased estimator default 1
    :return: standardized biased estimator
    """
    if not isinstance(data, pandas.DataFrame):
        raise TypeError("The type of input is wrong. Please use pandas.DataFrame.")
    return data.apply(lambda x: (x-np.mean(x))/np.std(x, ddof=ddof))


def reject_outliers(data=pandas.DataFrame(), min_thre=3, max_thre=6, filter_col=None):
    if not filter_col:
        filter_col = []

    for col in data.columns:
        if col in filter_col:
            continue
        data = data[~(operator.and_(min_thre * data[col].std() < np.abs(data[col]-data[col].mean()),
                      max_thre * data[col].std() > np.abs(data[col] - data[col].mean())))]
    return data
