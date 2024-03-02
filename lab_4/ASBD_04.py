# %% [markdown]
# # PROBLEM SET IV - DATA PREPROCESSING

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
# from scipy.stats import binned_statistic

# %% [markdown]
# # 1.
#
# Suppose that the data for analysis includes the attribute age. The age values
# for the data tuples are (in increasing order) 13, 15, 16, 16, 19, 20, 20, 21,
# 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70.
#
# (a) Use min-max normalization to transform the value 25 for age onto the
# range [0.0;1.0].
#
# (b) Use z-score normalization to transform the value 25 for age, where the
# standard deviation of age is 12.94 years.
#
# (c) Use normalization by decimal scaling to transform the value 25 for age
# such that transformed value is <1

# %%
age = [
    13,
    15,
    16,
    16,
    19,
    20,
    20,
    21,
    22,
    22,
    25,
    25,
    25,
    25,
    30,
    33,
    33,
    35,
    35,
    35,
    35,
    36,
    40,
    45,
    46,
    52,
    70,
]


# %%
# 1.a min max normalization
def min_max(v, new_min=0, new_max=1):
    old_min = min(age)
    old_max = max(age)
    return (v - old_min) / (old_max - old_min) * (
        new_max - new_min
    ) + new_min


min_max(25)


# %%
# 1.b z-score normalization
def z_score(v):
    mean = np.mean(age)
    # std = np.std(age)
    std = 12.94

    print("mean:", mean)
    print("std:", std)
    return (v - mean) / std


z_score(25)


# %%
# 1.c decimal scaling
def decimal_scaling(v, j):
    return v / 10**j


decimal_scaling(25, 2)

# %% [markdown]
# # 2.
#
# Use the given dataset and perform the operations listed below. Dataset
# Description It is a well-known fact that Millenials LOVE Avocado Toast. It's
# also a well known fact that all Millenials live in their parents basements.
# Clearly, they aren't buying home because they are buying too much Avocado
# Toast! But maybe there's hope... if a Millenial could find a city with cheap
# avocados, they could live out the Millenial American Dream. Help them to
# filter out the clutter using some pre-processing techniques.
#
# Some relevant columns in the dataset:
# - Date - The date of the observation
# - Average Price - the average price of a single avocado
# - type - conventional or organic
# - year - the year
# - Region - the city or region of the observation
# - Total Volume - Total number of avocados sold
# - 4046 - Total number of avocados with PLU* 4046 sold
# - 4225 - Total number of avocados with PLU* 4225 sold
# - 4770 - Total number of avocados with PLU* 4770 sold
#
# (Product Lookup codes (PLU’s)) *
#
# a. Sort the attribute “Total Volume” in the dataset given and distribute the
# data into equal sized/frequency bins of 50 & 250. Smooth the sorted data by
# (i) bin-means
# (ii) bin-medians
# (iii) bin-boundaries (smooth using bin boundaries after trimming the data
# by 2%).
#
# b. The dataset represents weekly retail scan data for National retail volume
# (units) and price. Retail scan data comes directly from retailers’ cash
# registers based on actual retail sales of Hass avocados. However, the company
# is interested in the monthly (total per month) and annual sales (total per
# year), rather than the total per week. So, reduce the data accordingly.
#
# c. Summarize the number of missing values for each attribute
#
# d. Populate data for the missing values of the attribute= “Average Price” by
# averaging their values that fall under the same region.
#
# e. Discretize the attribute=“Date” using concept hierarchy into {Old, New,
# Recent} {2015,2016 : Old, 2017: New, 2018: Recent} and plot in q-q plots

# %%
avocado = pd.read_csv("avocado_dataset.csv")
avocado = avocado.drop(columns=["4046", "4225", "4770", "type"])
avocado["AveragePrice"] = pd.to_numeric(
    avocado["AveragePrice"], errors="coerce"
)
print(avocado.describe())
avocado.head(10)


# %% [markdown]
# ### 2.a


# %%
def equal_depth_bins(array, bin_size):
    np.sort(array)
    bins = array.reshape((bin_size, -1))
    bmeans = []
    bmedians = []
    bboundaries = []
    for i in bins:
        mean = np.mean(i)
        median = np.median(i)
        for j in i:
            bmeans.append(mean)
            bmedians.append(median)
            if (j - i[0]) < (i[-1] - j):
                bboundaries.append(i[0])
            else:
                bboundaries.append(i[-1])

    avocado_tv["BinMeans_" + str(bin_size)] = bmeans
    avocado_tv["BinMedians_" + str(bin_size)] = bmedians
    avocado_tv["BinBoundary_" + str(bin_size)] = bboundaries


# %%
avocado_tv = pd.DataFrame()
equal_depth_bins(
    np.array([4, 8, 9, 15, 21, 21, 24, 25, 26, 28, 29, 34]), 3
)
avocado_tv

# %%
avocado_tv = pd.DataFrame()
equal_depth_bins(avocado["Total Volume"].to_numpy(), 50)
equal_depth_bins(avocado["Total Volume"].to_numpy(), 250)
avocado_tv

# %%
# np.histogram(avocado_tv, bins=3, weights=avocado_tv)[
# 0
# ] / np.histogram(avocado_tv, bins=3)[0]
# equal width bins
# bin_means, bin_edges, bin_numbers = binned_statistic(
#     avocado["Total Volume"],
#     avocado["Total Volume"],
#     statistic="mean",
#     bins=50,
# )
# x = np.array([4, 8, 9, 15, 21, 21, 24, 25, 26, 28, 29, 34])
# binned_statistic(
#      x, x, statistic="mean", bins=3
#  )
# bin_mean_values = []
# for i in bin_numbers:
#     bin_mean_values.append(bin_means[i - 1])
# bin_mean_values

# %% [markdown]
# ## 2.b

# %%
cols = [
    "Total Volume",
    "Total Bags",
    "Small Bags",
    "Large Bags",
    "XLarge Bags",
    "year",
]
avocado[cols].groupby(["year"]).sum()

# %%
avocado["month"] = pd.DatetimeIndex(avocado["Date"]).month
cols.append("month")
avocado[cols].groupby(["year", "month"]).sum()

# %% [markdown]
# ### 2.c Missing values for each attribute

# %%
avocado.shape[0] - avocado.count()

# %% [markdown]
# ### 2.d

# %%
print("mean:", avocado["AveragePrice"].mean())
avocado[["AveragePrice", "region"]].groupby("region").mean()

# %%
avocado = avocado.fillna(
    avocado[["AveragePrice", "region"]]
    .groupby("region")
    .transform("mean")
)
avocado.head(10)


# %% [markdown]
# ### 2.e


# %%
def discrete_year(year):
    if year == 2015 or year == 2016:
        return "Old"
    elif year == 2017:
        return "New"
    else:
        return "Recent"


dy = avocado["year"].map(discrete_year)
sm.qqplot(dy)
