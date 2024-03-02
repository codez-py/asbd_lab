# %% [markdown]
# # PROBLEM SET V - DATA PREPROCESSING

# %%
import pandas as pd
import numpy as np
from scipy import stats

# %%
avocado_original = pd.read_csv("./avocado_dataset.csv")
avocado_original["AveragePrice"] = pd.to_numeric(
    avocado_original["AveragePrice"], errors="coerce"
)
avocado_original

# %%
avocado_original.head(10)

# %% [markdown]
# # 1.
#
# Select a subset of relevant attributes from the given dataset that are
# necessary to know about the total volume of avocados with product lookup
# codes (PLU) 4046, 4225, 4770) which are of organic type. (Use AVOCADO
# dataset)

# %%
avocado = avocado_original.copy()
avocado = avocado[["Total Volume", "4046", "4225", "4770"]][
    avocado.type == "organic"
]
avocado


# %% [markdown]
# # 2.
#
# Discard all duplicate entries in the dataset given and fill all the missing
# values in the attribute “AveragePrice” as 1.25. Also print the size of the
# dataset before and after removing duplicates. (Use Trail dataset)

# %%
avocado = avocado_original.copy()
print("Before:", avocado.shape)
avocado.drop_duplicates(inplace=True)
print("After:", avocado.shape)
avocado["AveragePrice"].fillna(1.25, inplace=True)
avocado.head(10)

# %% [markdown]
# # 3.
#
# Binarize the attribute “Year”. Set the threshold above 2016 and print it
# without truncation. (Use AVOCADO dataset)

# %%
avocado = avocado_original.copy()
avocado.year = (avocado.year > 2016).astype(int)
avocado

# %% [markdown]
# # 4.
#
# Transform all categorical attributes in the dataset AVOCADO using Integer
# Encoding.


# %%
def integer_encode(df, field):
    df[field] = df[field].astype("category")
    df[field] = df[field].cat.codes


avocado = avocado_original.copy()
for column in ["type", "region"]:
    integer_encode(avocado, column)

avocado

# %% [markdown]
# # 5.
#
# Transform the attribute = “Region” in the given dataset AVOCADO using One-
# Hot Encoding.

# %%
avocado = avocado_original.copy()
print("Before:", avocado.shape)
encoded = pd.get_dummies(avocado, columns=["region"], dtype=int)
print("After:", encoded.shape)
encoded

# %%
encoded[["region_Albany", "region_WestTexNewMexico"]]

# %% [markdown]
# # 6.
#
# Ignore the tuples that hold missing values and print the subset of
# data from AVOCADO dataset excluding “NaN” values.

# %%
avocado = avocado_original.copy()
print("Before:", avocado.shape)
avocado.dropna(inplace=True)
print("After:", avocado.shape)
avocado

# %% [markdown]
# # 7.
#
# Drop the attribute that has high nullity as it facilitates efficient
# prediction. (Use AVOCADO dataset)

# %%
avocado = avocado_original.copy()
high_null_col = avocado.count().idxmin()
print("column with high nullity:", high_null_col)
avocado.drop(columns=high_null_col, inplace=True)
avocado

# %% [markdown]
# # 8.
#
# Study the entire dataset and report the complete statistical summary about
# the data (Use AVOCADO dataset)
#
# - Dimension of the dataset
# - Most frequently occurring value under every attribute.
# - Datatype of every attribute
# - Count
# - Mean
# - Standard Deviation
# - Minimum Value
# - Maximum value
# - 25% o
# - Median i.e. 50%
# - 75%
# - Find whether the class distribution of dataset is imbalanced. (Note: Fix
# the class label as “Type” in the
# given dataset)
# - Correlation matrix
# - Skewness of every attribute.

# %%
avocado = avocado_original.copy()
avocado.describe()

# %% [markdown]
# ### Dimension of the dataset

# %%
print("Dimensions:", avocado.shape)

# %% [markdown]
# ### Most frequently occurring value under every attribute.

# %%
avocado.mode()

# %% [markdown]
# ### Datatype of every attribute

# %%
avocado.dtypes

# %% [markdown]
# ### Count

# %%
avocado.count()

# %% [markdown]
# ### Mean

# %%
avocado.mean(numeric_only=True)

# %% [markdown]
# ### Standard Deviation

# %%
avocado.std(numeric_only=True)

# %% [markdown]
# ### Minimum Value

# %%
avocado.min()

# %% [markdown]
# ### Maximum Value

# %%
avocado.max()

# %% [markdown]
# ### 25%

# %%
avocado.quantile(0.25, numeric_only=True)

# %% [markdown]
# ### Median i.e. 50%

# %%
avocado.quantile(0.50, numeric_only=True)

# %% [markdown]
# ### 75%

# %%
avocado.quantile(0.75, numeric_only=True)

# %% [markdown]
# Find whether the class distribution of dataset is imbalanced. (Note: Fix the
# class label as “Type” in the given dataset)

# %%
avocado.type.value_counts()

# %% [markdown]
# The class distribution of attribute "Type" is balanced

# %% [markdown]
# ### Correlation matrix

# %%
avocado.corr(numeric_only=True)

# %% [markdown]
# ### Skewness of every attribute

# %%
avocado.skew(numeric_only=True)

# %% [markdown]
# (For the below exercises, you are free to choose an appropriate data set as
# merited by the problem statements)

# %% [markdown]
# # 9.
#
# Test drive the use of Gini Index, Information Gain, Entropy and other
# measures that are supported in your platform, performing the role of data
# selection.


# %%
def entropy_gini(df, field):
    labels = df[field]
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)

    entropy = -(probs * np.log2(probs)).sum()
    gini = 1 - (probs**2).sum()

    sorted = labels.sort_values()
    sp = sorted.shape[0] // 2
    left, right = sorted[:sp], sorted[sp:]
    e_before = stats.entropy(sorted.value_counts())
    e_left = stats.entropy(left.value_counts())
    e_right = stats.entropy(right.value_counts())

    e_split = (
        e_left * left.shape[0] + e_right * right.shape[0]
    ) / sorted.shape[0]

    information_gain = e_before - e_split

    print(field)
    print("entropy:", entropy)
    print("gini index:", gini)
    print("entropy after split:", e_split)
    print("information gain:", information_gain)
    print()


avocado = avocado_original.copy()
entropy_gini(avocado, "type")
entropy_gini(avocado, "year")
entropy_gini(avocado, "Total Volume")

# %% [markdown]
# # 10.
#
# Test drive the implementation support in your platform of choice for data
# preprocessing phases such as cleaning, selection, transformation, integration
# in addition to the earlier exercises.

# %%
df = pd.read_csv("laptopData.csv")
df

# %% [markdown]
#  ### selection

# %%
df.drop(columns="Unnamed: 0", errors="ignore", inplace=True)

# %% [markdown]
# ### transformation

# %%
df.Inches = pd.to_numeric(df.Inches, errors="coerce")
df.Ram = pd.to_numeric(df.Ram.str.rstrip("GB"), errors="coerce")
df.Weight = pd.to_numeric(df.Weight.str.rstrip("kg"), errors="coerce")

# %%
df.dtypes

# %%
df

# %%
print(df.shape)
df.describe()

# %% [markdown]
# ### cleaning

# %%
df["Inches"].fillna(
    df.groupby(["TypeName", "ScreenResolution"])["Inches"].transform(
        "median"
    ),
    inplace=True,
)
df["Weight"].fillna(
    df.groupby(["TypeName", "Memory", "Gpu"])["Weight"].transform(
        "mean"
    ),
    inplace=True,
)

# %%
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# %% [markdown]
# ### integration

# %%
extra = df[:30]
extra.loc[:, "Ram"] = extra.Ram * 2
extra.loc[:, "Price"] = extra.Price * 1.5
extra

# %%
df = pd.concat([df, extra], ignore_index=True)

# %%
print(df.shape)
df.describe()
