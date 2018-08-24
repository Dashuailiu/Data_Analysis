import pandas
import numpy


def z_score(data):
    """
    :param data
    :return: standardized
    """
    if not isinstance(data, pandas.DataFrame):
        raise TypeError("The type of input is wrong. Please use pandas.DataFrame.")
    return (data - data.mean())/data.std()


def z_score_np(data, ddof=1):
    """

    :param data:
    :param ddof: biased estimator default 1
    :return: standardized biased estimator
    """
    if not isinstance(data, pandas.DataFrame):
        raise TypeError("The type of input is wrong. Please use pandas.DataFrame.")
    return data.apply(lambda x: (x-numpy.mean(x))/numpy.std(x, ddof=ddof))
