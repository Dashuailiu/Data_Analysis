import pandas


def file_open(file_path):
    pandas.set_option("max_columns", 20)
    data = pandas.read_csv(file_path, header=0, sep=',', float_precision='round_trip')

    return data


if __name__ == '__main__':
    file_path = "./data/White Wine Dataset.csv"
    data = file_open(file_path)
    summary = data.describe()
    print(summary)
