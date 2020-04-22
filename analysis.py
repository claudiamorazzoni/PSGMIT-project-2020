import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# 1 - loading iris.csv
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)

# 2 - arranging the column data
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df.replace({'Iris-setosa': 'setosa'}, regex=True, inplace=True)
df.replace({'Iris-versicolor': 'versicolor'}, regex=True, inplace=True)
df.replace({'Iris-virginica': 'virginica'}, regex=True, inplace=True)

# 3 - basic statistics
# outputs a summary of each variable to a single text file
save_exit = sys.stdout
file = open('Summary.txt', 'w')
sys.stdout = file

# a) each class has the same number of instances
a = df[df['species'].str.contains("setosa")].groupby('species').size()
b = df[df['species'].str.contains("setosa")].groupby('species').size()
c = df[df['species'].str.contains("setosa")].groupby('species').size()
print(a)
print(b)
print(c)
# b) taking a look at the first 10 lines
w = df.head(10)
print(w)
# c) instances and attributes info
print(df.info())
# d)a little more detail
print(df.describe())
# e) What will be the correlation between the variables?
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(),
            annot = True,
            fmt = '.2f',
            cmap='Blues')
plt.title('Correlation between variables in the Iris dataset')
plt.savefig('Correlation.png')
plt.show()

sys.stdout = save_exit
file.close()


# 4 - box and whisker plots
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.savefig('Boxplot.png')
plt.show()

# saving a histogram of each variable to png files.
df.hist(column='petal_length', by='species', bins='auto')
plt.savefig('Petal_histogram.png')
df.hist(column='sepal_length', by='species', bins='auto')
plt.savefig('Sepal_histogram.png')
plt.show()
plt.close()

# outputs a scatter plot of each pair of variables - # another type of plot to see correlations
plt.figure()
plt.scatter(x="sepal_length", y="petal_width", label='Petal L&W', color='b', marker='d', s=10, data=df)
plt.xlabel('Sepal length')
plt.ylabel('Petal width')
plt.savefig('petalsepal_scatter.png')
plt.show()

plt.scatter(x="petal_length", y="petal_width", label='Petal L&W', marker='o', s=20, data=df)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.savefig('petallw_scatter.png')
plt.show()

plt.scatter(x="sepal_length", y="sepal_width", label='Sepal L&W', color='r', marker='h', s=10, data=df)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.savefig('sepallw_scatter.png')
plt.show()

plt.scatter(x="petal_length", y="sepal_width", label='Sepal L&W', color='k', marker='s', s=20, data=df)
plt.xlabel('Petal length')
plt.ylabel('Sepal width')
plt.savefig('sepalpetal_scatter.png')
plt.show()

pd.plotting.scatter_matrix(df, alpha=0.5, cmap='True', figsize=(12,10), diagonal='kde', marker='.')
plt.savefig('Iris_scattermatrix.png')
plt.show()

