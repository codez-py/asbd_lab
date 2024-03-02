# %%
import time
import pandas as pd
from math import ceil
from matplotlib import pyplot as plt
from efficient_apriori import itemsets_from_transactions, apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# %% [markdown]
# # Q1 Apriori
#
# Test drive the basic version of Apriori algorithm for Frequent Itemset
# Mining using the package / library support in the platform of your choice and
# generate frequent itemsets and association rules.
#
# a. Test it on a toy dataset to ensure the correctness of the algorithm.
# b. Test the same on a benchmark dataset FIMI workshop datasets
# (http://fimi.uantwerpen.be/data/) or other sources like Kaggle.


# %%
def ap(file_name, min_support, min_confidence):
    transactions = []
    with open("../datasets/" + file_name) as file:
        for line in file:
            transactions.append(line.strip().split(" "))

    return apriori(
        transactions,
        min_support=min_support,
        min_confidence=min_confidence,
    )


def table_freq_itemsets(itemsets):
    df_itemsets = []

    for key, values in itemsets.items():
        for s, support in values.items():
            df_itemsets.append([key, s, support])

    return pd.DataFrame(
        df_itemsets, columns=["Level", "Frequent Itemsets", "Support"]
    )


def table_rules(rules):
    df_rules = []

    for rule in rules:
        df_rules.append(
            [
                rule.lhs,
                rule.rhs,
                rule.count_lhs,
                rule.count_rhs,
                rule.confidence,
            ]
        )

    return pd.DataFrame(
        df_rules,
        columns=[
            "antecedents",
            "consequents",
            "SC(antecedents)",
            "SC(consequents)",
            "Confidence",
        ],
    )


# %%
t_itemsets, t_rules = ap("toy.txt", 2 / 9, 0.4)
table_freq_itemsets(t_itemsets)

# %%
table_rules(t_rules)

# %%
itemsets, rules = ap("T10I4D100K.dat", 0.01, 0.4)
table_freq_itemsets(itemsets)

# %%
table_rules(rules)


# %% [markdown]
# # Q3 FP-Growth
#
# Test drive the basic version of FP-Growth algorithm for Frequent Itemset
# Mining using the package / library support in the platform of your choice and
# generate frequent itemsets and association rules.
#
# a. Test it on a toy dataset to ensure the correctness of the algorithm.
# b. Test the same on a benchmark dataset FIMI workshop datasets
# (http://fimi.uantwerpen.be/data/) or other sources like Kaggle.


# %%
def fp(file_name, min_support, min_confidence):
    transactions = []
    with open("../datasets/" + file_name) as file:
        for line in file:
            transactions.append(line.strip().split(" "))

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    itemsets = fpgrowth(
        df, min_support=min_support, use_colnames=True
    )
    rules = association_rules(
        itemsets, metric="confidence", min_threshold=min_confidence
    )
    return itemsets, rules


# %%
t_itemsets, t_rules = fp("toy.txt", 2 / 9, 0.4)
t_itemsets

# %%
t_rules

# %%
itemsets, rules = fp("T10I4D100K.dat", 0.01, 0.4)
itemsets

# %%
rules

# %% [markdown]
# https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/
# association_rules

# ### Confidence
# confidence(A→C)=support(A→C) / support(A), range: [0,1]

# ### Lift
# lift(A→C)=confidence(A→C) / support(C), range: [0,∞]

# ### Leverage
# levarage(A→C) = support(A→C) − support(A) × support(C), range: [−1,1]

# ### Conviction
# conviction(A→C) = 1 − support(C) / 1 − confidence(A→C), range: [0,∞]

# ### Zhangs metric
# zhangs metric(A→C) = (confidence(A→C) − confidence(A'→C)) /
# Max[confidence(A→C), confidence(A'→C)], range: [−1,1]


# %% [markdown]
# # Q2,4,5 Compare Apriori & FP-Growth For various support
#
# 2. Test the Q1 algorithm with various support and confidence measures and
# generate a time comparison for varied data set sizes (minimum 8 comparisons).
# To do the performance comparison you may use benchmark datasets. Summarize
# the comparison details in a table format
#
# 4. Test the Q2 algorithm with various support and confidence measures and
# generate a time comparison for varied data set sizes (minimum 8 comparisons).
# To do the performance comparison you may use benchmark datasets. Summarize
# the comparison details in a table format.
#
# 5. Compare the time taken across the two algorithms: Apriori and FP-Growth
# algorithms under different combinations of support and confidence values.

# %%
files = {
    "T10I4D100K.dat": 0.01,
    "T40I10D100K.dat": 0.1,
    "mushroom.dat": 0.4,
    "retail.dat": 0.2,
}

# %%
overall_tt = []
x = range(20, 70, 2)

for file_name in files.keys():
    transactions = []
    with open("../datasets/" + file_name) as file:
        for line in file:
            transactions.append(line.strip().split(" "))

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    for s in x:
        start = time.perf_counter_ns()
        # Apriori
        itemsets_from_transactions(transactions, min_support=s / 100)
        mid = time.perf_counter_ns()
        # FP-Growth
        fis = fpgrowth(df, min_support=s / 100)
        end = time.perf_counter_ns()
        overall_tt.append(
            [
                file_name,
                s / 100,
                (mid - start) / 1e6,
                (end - mid) / 1e6,
            ]
        )

# %%
columns = ["file_name", "min_support", "apriori", "fp_growth"]
ott = pd.DataFrame(overall_tt, columns=columns)
ott

# %%
plt.figure(figsize=(15, 10))
rows = ceil(len(files) / 2)
i = 0
for file in files:
    i += 1
    plt.subplot(rows, 2, i)
    p1 = ott[ott.file_name == file]
    plt.plot(p1.min_support, p1.apriori, label="apriori")
    plt.plot(p1.min_support, p1.fp_growth, label="fp_growth")
    plt.xlabel("min_support")
    plt.ylabel("time_taken in ms")
    plt.title(file)
    plt.legend()


plt.show()

# %% [markdown]
# # Generate rules For various confidence

# %%
overall_ttc = []
cx = range(40, 100, 2)

for file_name, min_support in files.items():
    transactions = []
    with open("../datasets/" + file_name) as file:
        for line in file:
            transactions.append(line.strip().split(" "))

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    fis = fpgrowth(df, min_support=min_support)
    for c in cx:
        start = time.perf_counter_ns()
        association_rules(
            fis, metric="confidence", min_threshold=c / 100
        )
        end = time.perf_counter_ns()
        overall_ttc.append([file_name, c / 100, (end - start) / 1e6])

# %%
ottc = pd.DataFrame(
    overall_ttc, columns=["file_name", "min_confidence", "time_taken"]
)
ottc

# %%
plt.figure(figsize=(15, 10))
rows = ceil(len(files) / 2)
i = 0
for file in files:
    i += 1
    plt.subplot(rows, 2, i)
    p1 = ottc[ottc.file_name == file]
    plt.plot(
        p1.min_confidence, p1.time_taken, label="rules_generation"
    )
    plt.xlabel("min_confidence")
    plt.ylabel("time_taken in ms")
    plt.title(file)
    plt.legend()

plt.show()
