# %% [markdown]
# # PROBLEM SET III VISUALIZATION PLOTS

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from math import pi
from plotly import express as px

# %% [markdown]
# # 1.
# For a sample space of 15 people, a statistician wanted to know the
# consumption of water and other beverages. He collected their average
# consumption of water and beverages for 31 days (in litres). Help him to
# visualize the data using density plot, rug plot and identify the mean,
# median, mode and skewness of the data from the plot.

# |WATER|3.2|3.5|3.6|2.5|2.8|5.9|2.9|3.9|4.9|6.9|7.9|8.0|3.3|6.6|4.4|
# |-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
# |BEVERAGES|2.2|2.5|2.6|1.5|3.8|1.9|0.9|3.9|4.9|6.9|0.1|8.0|0.3|2.6|1.4|


# %%
def q1():
    data = pd.DataFrame(
        {
            "water": [
                3.2,
                3.5,
                3.6,
                2.5,
                2.8,
                5.9,
                2.9,
                3.9,
                4.9,
                6.9,
                7.9,
                8.0,
                3.3,
                6.6,
                4.4,
            ],
            "beverages": [
                2.2,
                2.5,
                2.6,
                1.5,
                3.8,
                1.9,
                0.9,
                3.9,
                4.9,
                6.9,
                0.1,
                8.0,
                0.3,
                2.6,
                1.4,
            ],
        }
    )

    data.plot.kde()
    sns.rugplot(data)

    plt.show()


q1()

# %% [markdown]
# The peak of density plot is the mode.

# %% [markdown]
# # 2.

# A car company wants to predict how much fuel different cars will use based on
# their masses. They took a sample of cars, drove each car 100km, and measured
# how much fuel was used in each case (in litres). Visualize the data using
# scatterplot and also find co-relation between the 2 variables (eg. Positive /
# Negative, Linear / Non-linear co-relation) The data is summarized in the
# table below.
#
# (Use a reasonable scale on both axes and put the explanatory variable on the
# x-axis.)

# |Fuel used (L)|3.6|6.7|9.8|11.2|14.7|
# |-|-|-|-|-|-|
# |Mass (metric tons)|0.45|0.91|1.36|1.81|2.27|


# %%
def q2():
    data = pd.DataFrame(
        {
            "fuel_used": [3.6, 6.7, 9.8, 11.2, 14.7],
            "mass": [0.45, 0.91, 1.36, 1.81, 2.27],
        }
    )

    data.plot(kind="scatter", x="mass", y="fuel_used")
    plt.axline(data.iloc[0][::-1], data.iloc[-1][::-1], color="red")
    plt.show()


q2()

# %% [markdown]
# As mass of vehicle increases the fuel_used also increases linearly.
#
# correlation: positive, linear

# %% [markdown]
# # 3.

# The data below represents the number of chairs in each class of a government
# high school. Create a box plot and swarm plot (add jitter) and find the
# number of data points that are outliers. 35, 54, 60, 65, 66, 67, 69, 70, 72,
# 73, 75, 76, 54, 25, 15, 60, 65, 66, 67, 69, 70, 72, 130, 73, 75, 76


# %%
def q3():
    chairs = [
        35,
        54,
        60,
        65,
        66,
        67,
        69,
        70,
        72,
        73,
        75,
        76,
        54,
        25,
        15,
        60,
        65,
        66,
        67,
        69,
        70,
        72,
        130,
        73,
        75,
        76,
    ]

    plt.subplot(121)
    plt.grid()
    plt.boxplot(chairs)
    plt.ylabel("chairs")

    chairs.extend([24, 110, 80, 70])

    plt.subplot(122)
    sns.swarmplot(chairs)
    plt.show()


q3()

# %% [markdown]
# 15, 25, 35, 130. These 4 points are outliers.

# %% [markdown]
# # 4.

# Generate random numbers from the following distribution and visualize the
# data using violin plot.

# - Standard-Normal distribution.
# - Log-Normal distribution.


# %%
def q4():
    standard_normal = np.random.standard_normal(1000)
    log_normal = np.random.lognormal(size=1000)

    plt.subplot(121)
    plt.violinplot(standard_normal)

    plt.subplot(122)
    plt.violinplot(log_normal)
    plt.show()


q4()

# %% [markdown]
# # 5.

# An Advertisement agency develops new ads for various clients (like Jewellery
# shops, Textile shops). The Agency wants to assess their performance, for
# which they want to know the number of ads they developed in each quarter
# for different shop category. Help them to visualize data using radar/spider
# charts.

# |Shop Category|Quarter 1|Quarter 2|Quarter 3|Quarter 4|
# |-|-|-|-|-|
# |Textile|10|6|8|13|
# |Jewellery|5|5|2|4|
# |Cleaning Essentials|15|20|16|15|
# |Cosmetics|14|10|21|11|


# %%
def spider_plot(df, i):
    categories = list(df)[1:]
    N = len(categories)

    values = df.loc[i].drop("Shop Category").values.flatten().tolist()
    values += values[:1]

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(2, 2, i + 1, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, "b", alpha=0.1)

    plt.xticks(angles[:-1], categories)
    plt.yticks(color="grey")
    plt.title(df.loc[i].iloc[0])


# %%
def q5():
    data = pd.DataFrame(
        {
            "Shop Category": [
                "Quarter 1",
                "Quarter 2",
                "Quarter 3",
                "Quarter 4",
            ],
            "Textile": [10, 6, 8, 13],
            "Jewellery": [5, 5, 2, 4],
            "Cleaning Essentials": [15, 20, 16, 15],
            "Cosmetics": [14, 10, 21, 11],
        }
    )

    plt.figure(figsize=(10, 10))
    for i in range(data.shape[0]):
        spider_plot(data, i)

    plt.show()


q5()

# %% [markdown]
# - More ads for textile in quarter 4.
# - ads for cleaning essentials in all quarters.
# - more ads for cosmetics in quarter 1 & 3.

# %% [markdown]
# # 6.

# An organization wants to calculate the % of time they spent on each process
# for their product development. Visualize the data using a funnel chart with
# the data given below.

# |Product Development steps|Time spent (in hours)|
# |-|-|
# |Requirement Elicitation|50|
# |Requirement Analysis|110|
# |Software Development|250|
# |Debugging & Testing|180|
# |Others|70|


# %%
def q6():
    data = pd.DataFrame(
        {
            "Product Dev Steps": [
                "Requirement Elicitaion",
                "Requirement Analysis",
                "Software Development",
                "Debugging & Testing",
                "Others",
            ],
            "Time Spent": [50, 110, 250, 180, 70],
        }
    )

    fig = px.funnel(
        data,
        y="Product Dev Steps",
        x="Time Spent",
        width=800,
        height=500,
    )
    fig.show(config={"staticPlot": True})


q6()

# %% [markdown]
# # 7.

# Let's say you are the new owner of a small ice-cream shop in a little village
# near the beach. You noticed that there was more business in the warmer months
# than the cooler months. Before you alter your purchasing pattern to match
# this trend, you want to be sure that the relationship is real. Help him to
# find the correlation between the data given.

# |Temperature|Number of Customers|
# |-|-|
# |98|15|
# |87|12|
# |90|10|
# |85|10|
# |95|16|
# |75|7|


# %%
def q7():
    data = pd.DataFrame(
        {
            "Temperature": [98, 87, 90, 85, 95, 75],
            "Number of Customers": [15, 12, 10, 10, 16, 7],
        }
    )

    data.plot(
        kind="scatter", x="Temperature", y="Number of Customers"
    )

    plt.axline(data.iloc[0], data.iloc[-1], color="red")
    plt.show()


q7()

# %% [markdown]
# As the temperature increses the number of customers buying ice creams
# increases linearly.

# %% [markdown]
# # 8.

# Given two arrays of numeric values, identify the suitable visualization
# mechanisms like Heat Maps to draw a relationship between the given two sets.
# Also extend the same for the IRIS dataset for possible attribute subset of
# your choice.
# Dataset: IRIS Dataset (https://www.kaggle.com/datasets/uciml/ iris)


# %%
def q8():
    iris_data = pd.read_csv("../Iris.csv")
    iris_data = iris_data.drop("Id", axis=1)

    correlation = iris_data.corr(numeric_only=True)

    sns.heatmap(
        correlation, annot=True, cmap="crest", linewidth=0.5
    )
    plt.show()

    return correlation


q8()

# %% [markdown]
# - The sepal length and the sepal width of the flower have a negative
# correlation since the correlation coefficient between these variables is
# negative
# - Sepal length and petal width are strongly correlated.
# - Petal length and petal width have a strong correlation.
# - Petal width and sepal width have a strong negative correlation
