from pylab import *

import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utilities import z_score
from Wine_quality import file_open, target_col


def data_overall(data=pandas.DataFrame()):
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


def visualization(data=pandas.DataFrame(), columns=None):
    for col_name in columns:
        sns.distplot(data[col_name])
        plt.xlabel(col_name)
        plt.show()


if __name__ == '__main__':
    data = file_open("./data/White Wine Dataset.csv")
    visualization(data, data.columns)
    data_overall(data)
