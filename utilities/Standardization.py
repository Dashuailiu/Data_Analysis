import pandas
import numpy


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
    return data.apply(lambda x: (x-numpy.mean(x))/numpy.std(x, ddof=ddof))
