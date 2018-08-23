import pandas

file_path = "./data/White Wine Dataset.csv"

data = pandas.read_csv(file_path, header=0, sep=',')

print(data.head())

summary = data.describe()
print(summary)

