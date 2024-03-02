# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MAPREDUCE & SPARK Programs

# %%
import numpy as np
import findspark
from math import exp, log
from pyspark.context import SparkContext
from pyspark.mllib.stat import Statistics
from pyspark.mllib.fpm import FPGrowth

# %%
findspark.init()
sc = SparkContext("local")

# %% [markdown]
# # 1
#
# For the given input file, calculate Wordcount using Hadoop MapReduce(optional) and
# Spark. Also, develop an equivalent conventional program (without spark RDDs) and
# compare the time taken by the two versions (one with spark and other without spark).

# %% [markdown]
# ## Using PySpark

# %%
wordcount_file = "../datasets/kosarak.dat"

text_file = sc.textFile(wordcount_file)
counts = (
    text_file.flatMap(lambda line: line.split(" "))
    .map(lambda word: (word, 1))
    .reduceByKey(lambda a, b: a + b)
)
output = counts.collect()

# printing only 10 most frequent words
for word, count in sorted(output, key=lambda x: x[1], reverse=True)[
    :10
]:
    print(f"{word}: {count}")

# %% [markdown]
# ## Using Normal Python

# %%
counter = {}

with open(wordcount_file) as file:
    for line in file:
        for word in line.strip().split(" "):
            if word not in counter:
                counter[word] = 0
            counter[word] += 1

# printing only 10 most frequent words
for word, count in sorted(
    counter.items(), key=lambda x: x[1], reverse=True
)[:10]:
    print(f"{word}: {count}")

# %% [markdown]
# # 2
#
# Randomly populate 1000 numbers and calculate mean, variance, standard deviation for
# the generated data.

# %%
np_arr = np.random.random(1000)
spark_arr = sc.parallelize(np_arr)

print("numpy:")
print("mean:", np_arr.mean())
print("variance:", np_arr.var())
print("standard deviation:", np_arr.std())
print()
print("pyspark:")
print("mean:", spark_arr.mean())
print("variance:", spark_arr.variance())
print("standard deviation:", spark_arr.stdev())

# %% [markdown]
# # 3
#
# Compute correlation between the given two series using Pearson’s and Spearman’s
# Method.
#
# a. (Use the Spark MLlib libraries and helper functions available)
# 1. Series A: 35, 23, 47, 17, 10, 43, 9, 6, 28
# 2. Series B: 30, 33, 45, 23, 8, 49, 12, 4, 31

# %%
seriesA = sc.parallelize([35, 23, 47, 17, 10, 43, 9, 6, 28])
seriesB = sc.parallelize([30, 33, 45, 23, 8, 49, 12, 4, 31])
print(
    "Correlation using pearson is:",
    Statistics.corr(seriesA, seriesB, method="pearson"),
)
print(
    "Correlation using spearman is:",
    Statistics.corr(seriesA, seriesB, method="spearman"),
)

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# # 4
#
# Randomly generate 10k numbers and apply the following functions on the generated
# random numbers. Develop two versions where in version-1 you will print all 10k
# numbers in the action statement whereas in the second version, print only 100 numbers
# using the action statement and compare the time taken by the two versions.
#
# 1. Exponential Function, f1(x)=ex
# 2. Logarithmic Function, f2(x)=log(x)

# %%
random_10k = np.random.random(10_000)
spark_10k = sc.parallelize(random_10k)

f1 = lambda x: exp(x)
f2 = lambda x: log(x)

e_x = spark_10k.map(f1).collect()
log_x = spark_10k.map(f2).collect()


print("\tx\t\te^x\t\tlog(x)")
np.array([random_10k, e_x, log_x]).T[:10]

# %% [markdown]
# # 5
#
# Generate FIMs using FPgrowth algorithm on pyspark setup using a benchmark dataset
# available on the FIMI website.

# %%
# sc = SparkContext(appName="FPGrowth")

data = sc.textFile("../datasets/mushroom.dat")
transactions = data.map(lambda line: line.strip().split(" "))
model = FPGrowth.train(transactions, minSupport=0.3, numPartitions=10)
result = model.freqItemsets().collect()

print("no. of frequent itemsets:", len(result))
for fi in result:
    print(fi)

# %%
sc.stop()
