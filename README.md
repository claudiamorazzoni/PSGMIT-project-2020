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

1 - I created a variable that receives the reading from the Iris csv file. (code - .read_csv)

2 - Observing the data at first, it was necessary to organize and rename the column names. (code - .columns and .replace)

3 - I started the basic statistics to have better data parameters. I captured the code outputs and saved it to a text file. (code - sy.stdout). I named this file 'Summary'.
  a) I could see that each class has the same number of instances (50 or 33% of the data set). I used the groupby () function, which basically divides the data into different groups according to the chosen variable (in this case, they were the names of each species).
  b) I analyzed the data from the first 10 lines to see if there was any clue of convergence or divergence of the information. ( code - .head)
  c) To have more details of the file about instances and attributes. (code - .info)
  d) I was able to verify that all numerical values have the same scale (centimeters) and similar intervals between 0 and 8 centimeters.(code - .describe) 
  e) What will be the correlation between the variables? (code - .corr)
  
4 - As the input variables are numeric, I was able to create box graphs for each one. This gives me a much clearer idea of the distribution of the input attributes.

5 - I was also able to create a histogram of each input variable to get an idea of the distribution. It appears that perhaps two of the input variables have a Gaussian distribution. This may be useful in the future to see how we can use algorithms that can exploit this assumption.

6 - With the scatterplot matrix I observed the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

## Install
This project requires Python 2.7 and the following Python libraries installed:

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
