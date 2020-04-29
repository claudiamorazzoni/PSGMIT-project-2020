# **PSGMIT-project-2020**

## About

#### This is an example of a notebook to demonstrate concepts of Data Science. The dataset is often used in data mining, classification and clustering examples and to test algorithms.

In 1936 Ronald Fisher, biologist and statistician presented a set of data. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species.

The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Just for reference, here are pictures of the three flowers species:

![iris-machinelearning](https://user-images.githubusercontent.com/29405430/79043073-c1b01300-7bf4-11ea-9038-e99a9785db50.png)
image from [Machine Learning] (https://www.datacamp.com/community/tutorials/machine-learning-in-r)

Four features were measured in centimeters from each sample: 
* the length of the sepals
* the width of the sepals
* the length of the petals
* the width of the petals

#### Information about the original paper and usages of the dataset can be found in the http://archive.ics.uci.edu/ml/datasets/Iris

## Code descriptions

1 - I created a variable that receives the reading from the Iris csv file. 
```python
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
```

2 - Observing the data at first, it was necessary to organize and rename the column names. 
```python
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df.replace({'Iris-setosa': 'setosa'}, regex=True, inplace=True)
df.replace({'Iris-versicolor': 'versicolor'}, regex=True, inplace=True)
df.replace({'Iris-virginica': 'virginica'}, regex=True, inplace=True)
```

3 - I started the basic statistics to have better data parameters. I captured the code outputs and saved it to a text file. I named this file 'Summary'.
  a) I could see that each class has the same number of instances (50 or 33% of the data set). I used the groupby () function, which basically divides the data into different groups according to the chosen variable (in this case, they were the names of each species).
  b) I analyzed the data from the first 10 lines to see if there was any clue of convergence or divergence of the information. 
  c) To have more details of the file about instances and attributes. 
  d) I was able to verify that all numerical values have the same scale (centimeters) and similar intervals between 0 and 8 centimeters. 
  e) What will be the correlation between the variables? 
  ```python
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
```
 
4 - As the input variables are numeric, I was able to create box graphs for each one. This gives me a much clearer idea of the distribution of the input attributes.

```python
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.savefig('Boxplot.png')
plt.show()
```

5 - I was also able to create a histogram of each input variable to get an idea of the distribution. It appears that perhaps two of the input variables have a Gaussian distribution. This may be useful in the future to see how we can use algorithms that can exploit this assumption.

```python
df.hist(column='petal_length', by='species', bins='auto')
plt.savefig('Petal_histogram.png')
df.hist(column='sepal_length', by='species', bins='auto')
plt.savefig('Sepal_histogram.png')
plt.show()
plt.close()
```

6 - With the scatterplot matrix I observed the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship. The value shown for each correlation ranges from -1 - which indicates a perfect negative correlation - to +1 - a perfect positive correlation. Therefore, the closer to 1 or -1 there is a strong correlation.

```python
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
```


![Correlation](https://user-images.githubusercontent.com/29405430/79970153-68fd3780-848a-11ea-9e9b-178e136b1dbd.png)


## Install
This project requires Python 3 and the following Python libraries installed:

* Pandas
* Matplotlib


## References
* https://archive.ics.uci.edu/ml/datasets/iris
* https://www.makeareadme.com/
* https://en.wikipedia.org/wiki/Iris_flower_data_set
* https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html
* https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.hist.html
* https://matplotlib.org/3.1.3/api/_as_gen/matplotlib.pyplot.scatter.html
* https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html
* https://www.kaggle.com/gopaltirupur/iris-data-analysis-and-machine-learning-python
