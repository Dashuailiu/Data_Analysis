from pylab import *

import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import kstest, normaltest

from utilities import z_score, z_score_np


def file_open():
    file_path = "./data/White Wine Dataset.csv"

    pandas.set_option("max_columns", 20)
    data = pandas.read_csv(file_path, header=0, sep=',')

    return data


def data_overall(data=pandas.DataFrame()):
    target_col = 'quality'
    print(data.dtypes)
    print(data.head())

    summary = data.describe()
    print(summary)

    std_data = z_score(data, [target_col])
    print(std_data)

    # box plot for each attributes
    plt.boxplot(std_data.values)
    plt.xlabel("Attributes")
    plt.ylabel("Quartile Range - Normalized")
    plt.show()

    cmp = plt.cm.RdYlBu
    col_num = len(std_data.columns)-1
    r_num = len(std_data.index)
    legend_list = list()
    legend_patch_list = list()
    target_mean = np.mean(data[target_col])
    target_std = np.std(data[target_col])
    std_target_func = lambda x: (x - target_mean)/target_std

    for i in range(r_num):
        record = std_data.iloc[i, 1:col_num]
        label_color = 1.0/(1.0 + exp(-std_target_func(std_data.iloc[i, col_num])))
        if std_data.iloc[i, col_num] not in legend_list:
            legend_patch = mpatches.Patch(color=cmp(label_color), label=std_data.iloc[i, col_num])
            legend_patch_list.append(legend_patch)
            legend_list.append(std_data.iloc[i, col_num])
        record.plot(color=cmp(label_color), alpha=0.5, label=std_data.iloc[i, col_num])

    plt.xlabel("Attribute Index")
    plt.ylabel("Attribute Values")
    legend_patch_list.sort(key=lambda x: x.get_label())
    plt.legend(handles=legend_patch_list)
    plt.show()


def visualization(func, data=pandas.DataFrame(), columns=None):
    for col_name in columns:
        func(data[col_name])
        plt.xlabel(col_name)
        plt.show()


if __name__ == '__main__':
    data = file_open()
    visualization(plt.hist, data, data.columns)
    data_overall(data)
