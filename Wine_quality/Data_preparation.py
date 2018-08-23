import pandas


def pre_processing():
    file_path = "./data/White Wine Dataset.csv"

    pandas.set_option("max_columns", 20)
    data = pandas.read_csv(file_path, header=0, sep=',')

    print(data.dtypes)
    print(data.head())

    summary = data.describe()
    print(summary)

    std_data = standardization(data, summary)
    print(std_data)


def standardization(data, summary):
    """
    :param data
    :param summary count, mean, standard deviation
    :return: standized
    """
    for i in range(data.shape[1]):
        mean = summary.iloc[1, i]
        std = summary.iloc[2, i]
        data.iloc[:, i] = (data.iloc[:, i] - mean) / std

    return data


if __name__ == '__main__':
    pre_processing()
