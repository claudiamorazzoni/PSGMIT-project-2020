import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading iris.csv
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
print(df)

# basic statistics - outputs a summary of each variable to a single text file
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(df.columns)
df.replace({'Iris-setosa': 'setosa'}, regex=True, inplace=True)
df.replace({'Iris-versicolor': 'versicolor'}, regex=True, inplace=True)
df.replace({'Iris-virginica': 'virginica'}, regex=True, inplace=True)

# 1 - take a look the first 10 rows of data for each class
print(df.head(10))

# 2 - Division of `Species`
a = df[df['species'].str.contains("setosa")].groupby('species').size()
b = df[df['species'].str.contains("setosa")].groupby('species').size()
c = df[df['species'].str.contains("setosa")].groupby('species').size()
print(a)
print(b)
print(c)
# 3 - Summary
print(df.info())
#
# calculando o histograma para uma variável

df.hist(column='petal_length', by='species', bins='auto')
df.hist(column='sepal_length', by='species', bins='auto')
plt.show()
plt.close()
