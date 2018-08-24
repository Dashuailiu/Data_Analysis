import pandas

from pylab import *
import matplotlib.pyplot as plt

from utilities import z_score, z_score_np


def data_overall():
    file_path = "./data/White Wine Dataset.csv"

    pandas.set_option("max_columns", 20)
    data = pandas.read_csv(file_path, header=0, sep=',')

    print(data.dtypes)
    print(data.head())

    summary = data.describe()
    print(summary)

    std_data = z_score(data)
    print(std_data)

    # plt.boxplot(std_data.values)
    # plt.xlabel("Attributes")
    # plt.ylabel("Quartile Range - Normalized")
    # plt.show()

    cmp = plt.cm.RdYlBu
    col_num = len(std_data.columns)-1
    r_num = len(std_data.index)
    for i in range(r_num):
        record = std_data.iloc[i, 1:col_num]
        label_color = 1.0/(1.0 + exp(-std_data.iloc[i, col_num]))
        record.plot(color=cmp(label_color), alpha=0.5, label=std_data.iloc[i, col_num])

    plt.xlabel("Attribute Index")
    plt.ylabel("Attribute Values")
    labels = plt.get_figlabels()
    plt.legend(labels[::-1])
    plt.show()


if __name__ == '__main__':
    data_overall()
