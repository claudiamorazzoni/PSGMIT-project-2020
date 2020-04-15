import pandas as pd
import sys
import matplotlib.pyplot as plt


# loading iris.csv
d= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df = pd.DataFrame(d)
print(df)

# basic statistics - outputs a summary of each variable to a single text file
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(df.columns)
df.replace({'Iris-setosa': 'setosa'}, regex=True, inplace=True)
df.replace({'Iris-versicolor': 'versicolor'}, regex=True, inplace=True)
df.replace({'Iris-virginica': 'virginica'}, regex=True, inplace=True)

# a - take a look the first 10 rows of data for each class
save_exit = sys.stdout
file = open('Summary.txt', 'w')
sys.stdout = file
w = df.head(10)
print(w)
print(df.info())
sys.stdout = save_exit
file.close()

# b - Division of `Species`
a = df[df['species'].str.contains("setosa")].groupby('species').size()
b = df[df['species'].str.contains("setosa")].groupby('species').size()
c = df[df['species'].str.contains("setosa")].groupby('species').size()
print(a)
print(b)
print(c)

# c - Summary
print(df.info())

# saves a histogram of each variable to png files,

df.hist(column='petal_length', by='species', bins='auto') # These show that the distribution of values for Petal.Length are different for each class.
plt.savefig('Petal_histogram.png')
df.hist(column='sepal_length', by='species', bins='auto')
plt.savefig('Sepal_histogram.png')
plt.show()
plt.close()

# # outputs a scatter plot of each pair of variables
# another type of plot to see correlations
#g = sns.lmplot(x='petal_length', y='sepal_length', hue='species', data=df, palette='Set1')

#plt.show()
plt.scatter(x="petal_length", y="petal_width", label='Petal L&W', color='g', marker='*', s=100, data=df)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.savefig('petal_scatter.png')
plt.show()
plt.scatter(x="sepal_length", y="sepal_width", label='Sepal L&W', color='r', marker='*', s=100, data=df)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.savefig('sepal_scatter.png')
plt.show()