import pandas
import matplotlib.pyplot as plt


from pylab import *

from utilities import z_score, z_score_np


def pre_processing():
    file_path = "./data/White Wine Dataset.csv"

    pandas.set_option("max_columns", 20)
    data = pandas.read_csv(file_path, header=0, sep=',')

    print(data.dtypes)
    print(data.head())

    summary = data.describe()
    print(summary)

    std_data = z_score(data)
    print(std_data)

    plt.boxplot(std_data.values)
    plt.xlabel("Attributes")
    plt.ylabel("Quartile Range - Normalized")
    plt.show()


if __name__ == '__main__':
    pre_processing()
