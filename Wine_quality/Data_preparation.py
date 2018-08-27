from pylab import *
from utilities import reject_outliers
from Wine_quality import file_open, target_col

import seaborn as sns
import matplotlib.pyplot as plt


def pre_processing(data):
    print(data.head())
    summary = data.describe()
    print(summary)

    # Missing values
    for col in data.columns:
        if data[col].dtype == 'object':
            continue
        if data[col].count() != len(np.nonzero(data[col])):
            data[col].fillna(data[col].mean(), inplace=True)

    plt.figure(figsize=(18, 6))
    sns.boxplot(data=data)
    plt.xlabel("Attributes")
    plt.ylabel("Quartile Range - Normalized")
    plt.show()

    # remove duplicated records
    data.drop_duplicates(inplace=True)

    # remove outliers but keep extreme values
    data = reject_outliers(data, 2, 6, [target_col])

    plt.figure(figsize=(18, 6))
    sns.boxplot(data=data)
    plt.xlabel("Attributes")
    plt.ylabel("Quartile Range - Normalized & Remove Outlier")
    plt.show()

    # reclassification
    # (3.0, 9.0) => poor, fair, good, excellent
    rec_dict = {
        3: 'poor',
        4: 'poor',
        5: 'fair',
        6: 'good',
        7: 'good',
        8: 'excellent',
        9: 'excellent'
    }
    data[target_col] = data[target_col].map(rec_dict)
    print(data.head())
    sns.countplot(x=target_col, data=data)
    plt.show()
    print(data.describe(include='all'))

    data.to_csv("./data/clean_data.csv", columns=data.columns, index=None)


if __name__ == '__main__':
    data = file_open("./data/White Wine Dataset.csv")
    pre_processing(data)
