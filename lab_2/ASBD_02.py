# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + id="dYZNOH264Q6P"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# import stemgraphic

# + [markdown] id="oxSuXRmF4azm"
# # 1.
#
# Consider a subset of attributes in the given dataset and apply the below given visualization plots in
# any platform of your choice (Microsoft Excel / LibreOffice Calc / Python(preferred) / R).
#
# 1. Bar/Column Chart
# 2. Pie Chart
# 3. Scatter plot
# 4. Line Chart
# 5. Histogram
#
# Dataset: IRIS Dataset (https://www.kaggle.com/datasets/uciml/iris)
#
# It includes three iris species with 50 samples each as well as some properties about each flower. One
# flower species is linearly separable from the other two, but the other two are not linearly separable from
# each other.
#
# The columns in this dataset are:
#
# - Id
# - SepalLengthCm
# - SepalWidthCm
# - PetalLengthCm
# - PetalWidthCm
# - Species

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="Vm545J7U4cDa" outputId="2e9c0ca0-c281-4193-b72e-16e825979819"
iris_data = pd.read_csv('Iris.csv')

setosa = iris_data[iris_data.Species == 'Iris-setosa']
versicolor = iris_data[iris_data.Species == 'Iris-versicolor']
virginica = iris_data[iris_data.Species == 'Iris-virginica']

color = {
    'Iris-setosa': 'royalblue',
    'Iris-versicolor': 'darkorange',
    'Iris-virginica': 'limegreen',
}

iris_data


# + colab={"base_uri": "https://localhost:8080/", "height": 850} id="9aTiGSIz8OXo" outputId="58e93a88-5f57-45b4-9176-7b6416463e43"
# 1.1 Bar/Column chart

def bar_plot(x, xlabel):
  heights, bins = np.histogram(x, density = True, bins = 10)
  bin_width = np.diff(bins)[0]
  bin_pos =( bins[:-1] + bin_width / 2)
  plt.bar(bin_pos, heights, width = bin_width, edgecolor = 'black')
  plt.xlabel(xlabel)
  plt.ylabel('Frequency')


plt.figure(figsize = (10, 10))

plt.subplot(221)
bar_plot(iris_data.SepalLengthCm, 'Sepal Length in Cm')

plt.subplot(222)
bar_plot(iris_data.SepalWidthCm, 'Sepal Width in Cm')

plt.subplot(223)
bar_plot(iris_data.PetalLengthCm, 'Petal Length in Cm')

plt.subplot(224)
bar_plot(iris_data.PetalWidthCm, 'Petal Width in Cm')

plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 445} id="xRk7v4lr4fZT" outputId="018e52da-c58f-47c8-bb02-641bd37df964"
unique_species, counts = np.unique(iris_data['Species'], return_counts = True)

plt.figure(figsize = (10, 5))
plt.subplot(121)
plt.bar(unique_species, counts, color = color.values() )
plt.ylabel('count')

# 1.2. Pie chart
plt.subplot(122)
plt.pie(counts, labels = unique_species, autopct = '%.2f%%', colors = color.values())
plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="-XQIEoDT4juX" outputId="b2e5ecdd-ce12-4dd1-847e-175147f1d733"
# 1.3 Scatter plot

colors = [ color[species] for species in iris_data.Species ]

plt.figure(figsize = (15, 15))

plt.subplot(331)
plt.scatter(iris_data.SepalLengthCm, iris_data.SepalWidthCm, c = colors)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.subplot(332)
plt.scatter(iris_data.SepalLengthCm, iris_data.PetalWidthCm, c = colors)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')

plt.subplot(333)
plt.scatter(iris_data.SepalLengthCm, iris_data.PetalLengthCm, c = colors)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')

plt.subplot(334)
plt.scatter(iris_data.SepalWidthCm, iris_data.SepalLengthCm, c = colors)
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')

plt.subplot(335)
plt.scatter(iris_data.SepalWidthCm, iris_data.PetalWidthCm, c = colors)
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')

plt.subplot(336)
plt.scatter(iris_data.SepalWidthCm, iris_data.PetalLengthCm, c = colors)
plt.xlabel('Sepal Width')
plt.ylabel('Petal Length')

plt.subplot(337)
plt.scatter(iris_data.PetalLengthCm, iris_data.SepalWidthCm, c = colors)
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')

plt.subplot(338)
plt.scatter(iris_data.PetalLengthCm, iris_data.PetalWidthCm, c = colors)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.subplot(339)
plt.scatter(iris_data.PetalLengthCm, iris_data.SepalLengthCm, c = colors)
plt.xlabel('Petal Length')
plt.ylabel('Sepal Length')

plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 465} id="Q_YG1Uzd6kA-" outputId="880d3d9f-0477-4607-ce94-7760ada3aa84"
# 1.4 Line chart

plt.figure(figsize = (15, 5))

plt.plot(iris_data.Id, iris_data.SepalWidthCm, label = 'Sepal Width')
plt.plot(iris_data.Id, iris_data.PetalWidthCm, label = 'Petal Width')
plt.plot(iris_data.Id, iris_data.PetalLengthCm, label = 'Petal Length')
plt.plot(iris_data.Id, iris_data.SepalLengthCm, label = 'Petal Width')
plt.xlabel('Id')
plt.legend()

plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 850} id="ccXvHJ0Z4kiE" outputId="f6c794a2-71f9-40d1-a875-3ff124db6c5b"
# 1.5 Histogram

plt.figure(figsize = (10, 10))

plt.subplot(221)
plt.hist(iris_data.SepalLengthCm)
plt.xlabel('SepalLength')
plt.ylabel('Frequency')

plt.subplot(222)
plt.hist(iris_data.SepalWidthCm, bins = 8)
plt.xlabel('SepalWidth')
plt.ylabel('Frequency')


plt.subplot(223)
plt.hist(iris_data.PetalLengthCm)
plt.xlabel('PetalLength')
plt.ylabel('Frequency')

plt.subplot(224)
plt.hist(iris_data.PetalWidthCm, bins = 5)
plt.xlabel('PetalWidth')
plt.ylabel('Frequency')

plt.show()

# + colab={"base_uri": "https://localhost:8080/", "height": 789} id="j5q8sirDRkLM" outputId="a0bdf377-db4b-418e-b86f-ed78136efcd4"
plt.figure(figsize = (30, 15))

plt.subplot(341)
plt.hist(setosa.PetalLengthCm, label = 'Iris-setosa', color = color['Iris-setosa'])
plt.xlabel('Petal Length')
plt.legend()

plt.subplot(342)
plt.hist(setosa.PetalWidthCm, label = 'Iris-setosa', color = color['Iris-setosa'])
plt.xlabel('Petal Width')
plt.legend()

plt.subplot(343)
plt.hist(setosa.SepalLengthCm, label = 'Iris-setosa', color = color['Iris-setosa'])
plt.xlabel('Sepal Length')
plt.legend()

plt.subplot(344)
plt.hist(setosa.SepalWidthCm, label = 'Iris-setosa', color = color['Iris-setosa'])
plt.xlabel('Sepal Width')
plt.legend()

plt.subplot(345)
plt.hist(versicolor.PetalLengthCm, label = 'Iris-versicolor', color = color['Iris-versicolor'])
plt.xlabel('Petal Length')
plt.legend()

plt.subplot(346)
plt.hist(versicolor.PetalWidthCm, label = 'Iris-versicolor', color = color['Iris-versicolor'])
plt.xlabel('Petal Width')
plt.legend()

plt.subplot(347)
plt.hist(versicolor.SepalLengthCm, label = 'Iris-versicolor', color = color['Iris-versicolor'])
plt.xlabel('Sepal Length')
plt.legend()

plt.subplot(348)
plt.hist(versicolor.SepalWidthCm, label = 'Iris-versicolor', color = color['Iris-versicolor'])
plt.xlabel('Sepal Width')
plt.legend()

plt.subplot(349)
plt.hist(virginica.PetalLengthCm, label = 'Iris-virginica', color = color['Iris-virginica'])
plt.xlabel('Petal Length')
plt.legend()

plt.subplot(3,4, 10)
plt.hist(virginica.PetalWidthCm, label = 'Iris-virginica', color = color['Iris-virginica'])
plt.xlabel('Petal Width')
plt.legend()

plt.subplot(3, 4, 11)
plt.hist(virginica.SepalLengthCm, label = 'Iris-virginica', color = color['Iris-virginica'])
plt.xlabel('Sepal Length')
plt.legend()

plt.subplot(3, 4, 12)
plt.hist(virginica.SepalWidthCm, label = 'Iris-virginica', color = color['Iris-virginica'])
plt.xlabel('Sepal Width')
plt.legend()

plt.show()


# + [markdown] id="LlPisyug4prA"
# # 2.
#
# A Coach tracked the number of points that each of his 30 players on the team had in one game. The
# points scored by each player is given below. Visualize the data using ordered stem-leaf plot and also
# detect the outliers and shape of the distribution.
# 22, 21, 24, 19, 27, 28, 24, 25, 29, 28, 26, 31, 28, 27, 22, 39, 20, 10, 26, 24, 27, 28, 26, 28, 18, 32, 29,
# 25, 31, 27.

# + colab={"base_uri": "https://localhost:8080/", "height": 449} id="JkPQjtLb4qfK" outputId="0b71031a-1a93-4696-dc48-1b1139d4aeae"
def q2():
    scores = [ 22, 21, 24, 19, 27, 28, 24, 25, 29, 28, 26, 31, 28, 27, 22, 39, 20, 10, 26, 24, 27, 28, 26, 28, 18, 32, 29,
              25, 31, 27 ]

    stems = map(lambda x: x // 10, scores)
    leaves = map(lambda x: x % 10, scores)

    plt.stem([*stems], [*leaves])

    plt.xlabel('stems')
    plt.ylabel('score')
    plt.xlim(0, 5)
    plt.legend(['2|6 is 26'])
    plt.show()

    # stemgraphic.stem_graphic(scores, scale = 10)

q2()


# + [markdown] id="QQiaSA704tU8"
# 3.
#
# On New Year’s Eve, Tina walked into a random shop and surprised to see a huge crowd there. She
# is interested to find what kind of products they sell the most, for which she needs the age distribution
# of customers. Help her to find out the same using histogram. The age details of the customers are given
# below
# 7, 9, 27, 28, 55, 45, 34, 65, 54, 67, 34, 23, 24, 66, 53, 45, 44, 88, 22, 33, 55, 35, 33, 37, 47, 41,31, 30,
# 29, 12.

# + colab={"base_uri": "https://localhost:8080/", "height": 449} id="hCCNulEo4wwZ" outputId="f590136e-9fcc-4430-aa33-b9b9409ddac1"
def q3():
    age = [ 7, 9, 27, 28, 55, 45, 34, 65, 54, 67, 34, 23, 24, 66, 53, 45, 44, 88, 22, 33, 55, 35, 33, 37, 47, 41,31, 30,
       29, 12 ]
    plt.hist(age)
    plt.xlabel('age')
    plt.ylabel('count')
    plt.show()

q3()
