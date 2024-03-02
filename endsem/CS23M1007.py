# %% [markdown]
# # Analytics and Systems of Big Data Practice Midsem
# ## CS23M1007 VIMALRAJ S

# %%
import numpy as np
import pandas as pd
from math import pi
from scipy import stats
from collections import defaultdict
from itertools import chain, combinations
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

pd.set_option("display.max_rows", 120)

# %% [markdown]
# # FIFA 21 complete player dataset
#
# https://www.kaggle.com/stefanoleone992/fifa-21-complete-player-dataset

# %%
players = pd.read_csv("./dataset/players_21.csv")
print("shape:", players.shape)
players.describe()

# %% [markdown]
# # Midsem

# %% [markdown]
# ### DataType & No. of unique items

# %%
pd.DataFrame(
    [
        (c, players[c].dtype, len(players[c].unique()))
        for c in players.columns
    ],
    columns=["column", "dtype", "no. of unique_items"],
)

# %% [markdown]
# ### Count of Null values

# %%
null_values = players.isnull().sum()
null_values[null_values > 0]

# %% [markdown]
# ## Data Preprocessing
#
# ## Data Cleaning
#
# ### Filling null values with median
#
# For some of these values like shooting, passing, dribbling, defending, pace, physic
# are null only for three team_positions (SUB, GK, RES) in 2083 records.
#
# I am replacing the null values with the median values of their respective clubs.

# %%
for attribute in [
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "pace",
    "physic",
]:
    print(attribute)
    print(
        players[
            players[attribute].isnull()
        ].team_position.value_counts()
    )
    print()
    players[attribute].fillna(
        players.groupby(["club_name"])["shooting"].transform(
            "median"
        ),
        inplace=True,
    )

# %%
# before row 3 was null
# replaced null value with median
players[
    [
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "pace",
        "physic",
    ]
][:5]

# %% [markdown]
# ### Dropping Records
#
# club_name, league_name, league_rank, team_position, team_jersey_number and contract_valid_until
# attributes are null for 225 records.
#
# These 225 players are not in any league or club.
# I will drop these records and focus only on players who are in a club and league

# %%
print("before:", players.shape)
players = players[
    players[
        [
            "club_name",
            "league_name",
            "league_rank",
            "team_position",
            "team_jersey_number",
            "contract_valid_until",
        ]
    ]
    .notnull()
    .any(axis=1)
]
print("after:", players.shape)

# %%
# If more than half the values are null drop the column
to_drop = []
for c in players.columns:
    count = players[c].count()
    if count < (players.shape[0] // 2):
        to_drop.append(c)

players.drop(columns=to_drop)

to_drop

# %% [markdown]
# ### Remove duplicates

# %%
players.drop_duplicates(inplace=True)

# %% [markdown]
# ## Data Visualization

# %% [markdown]
# ### Pie chart

# %% [markdown]
# #### Preferred foot for players

# %%
pfoot, counts = np.unique(players.preferred_foot, return_counts=True)
plt.pie(counts, labels=pfoot, autopct="%0.2f%%")
plt.legend()
plt.show()

# %% [markdown]
# ## Bar Chart

# %%
plt.figure(figsize=(10, 5))
nationality, counts = np.unique(
    players.nationality, return_counts=True
)
plt.bar(nationality[10:20], counts[10:20])
plt.show()

# %% [markdown]
# ### Histogram, Box, Violin plot

# %% [markdown]
# #### Age distribution of all players

# %%
mean = players.age.mean()
median = players.age.median()
mode = stats.mode(players.age).mode

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.hist(players.age)
plt.axvline(players.age.mean(), label="mean", color="red")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(2, 1, 2)
plt.boxplot(players.age, vert=False, patch_artist=True)
plt.violinplot(players.age, vert=False)
plt.xlabel("Age")
plt.show()

# %% [markdown]
# #### Height distribution of all players

# %%
mean = players.height_cm.mean()
median = players.height_cm.median()
mode = stats.mode(players.height_cm).mode

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.hist(players.height_cm, bins=15)
plt.axvline(players.height_cm.mean(), label="mean", color="red")
plt.xlabel("Height in cm")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(2, 1, 2)
plt.boxplot(players.height_cm, vert=False, patch_artist=True)
plt.violinplot(players.height_cm, vert=False)
plt.xlabel("Height in cm")
plt.show()

# %% [markdown]
# #### Weight distribution of all players

# %%
mean = players.weight_kg.mean()
median = players.weight_kg.median()
mode = stats.mode(players.weight_kg).mode

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.hist(players.weight_kg, bins=15)
plt.axvline(players.weight_kg.mean(), label="mean", color="red")
plt.xlabel("Weight in cm")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(2, 1, 2)
plt.boxplot(players.weight_kg, vert=False, patch_artist=True)
plt.violinplot(players.weight_kg, vert=False)
plt.xlabel("Weight in Kg")
plt.show()

# %% [markdown]
# ### Scatter plots
#
# #### Comparing Height & Weight

# %%
plt.scatter(players.height_cm, players.weight_kg)
plt.xlabel("Height in cm")
plt.ylabel("Weight in Kg")
plt.show()

# %% [markdown]
# #### Comparing potential and value

# %%
players.plot(kind="scatter", x="potential", y="value_eur")
plt.show()

# %% [markdown]
# As the potential of player increases the value (in EUR) increases expoentially

# %% [markdown]
# ### Box plot

# %% [markdown]
# #### Comparing Age between different clubs

# %%
x = players.club_name.unique()[:10]
plt.figure(figsize=(20, 10))

for i, c in enumerate(x):
    plt.boxplot(
        players[players.club_name == c].age,
        positions=[i],
        widths=0.5,
        labels=[c],
    )

plt.show()


# %% [markdown]
# - In boxes for Paris Saint-Germain and Manchester City, the upper quartile has low spread
# whereas the lower quartile has more spread.
# - For FC Bayem Munchen the upper quartile has more spread and lower quartile has less spread.
# - The median (Q2) for Real Madrid and Tottenham Hotspur are overlapping,
# and the boxes are also overlapping. They are more similar data in those two.

# %% [markdown]
# ### Spider / Radar Plot

# %% [markdown]
# #### Comparing different players


# %%
def spider_plot(df, i):
    N = len(df.columns) - 1

    values = df.loc[i][1:].tolist()  # remove name from plotting
    values += values[:1]

    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    ax = plt.subplot(2, 2, i + 1, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, "b", alpha=0.1)

    plt.xticks(angles[:-1], df.columns[1:])
    plt.yticks(color="grey")
    plt.ylim(0, 100)
    plt.title(df.short_name.loc[i])


# %%
spider_columns = players[
    [
        "short_name",
        "shooting",
        "passing",
        "dribbling",
        "defending",
        "pace",
        "physic",
    ]
]
plt.figure(figsize=(10, 10))
for i in range(4):
    spider_plot(spider_columns, i)


# %% [markdown]
# - J. Oblak is performing average in all categories. This is one of missing data replaced using median
# - All others are good in shooting, they are bad at defending.
# They are mostly playing as forwards

# %% [markdown]
# ## FP-Growth

# %% [markdown]
# ### Data selection


# %%
# will return the range the given data is
def bin_b(x, column, boundaries):
    for j in range(1, len(boundaries)):
        if x < boundaries[j]:
            return "%d-%d" % (
                int(boundaries[j - 1]),
                int(boundaries[j]),
            )

    return f">{int(boundaries[-1])}"


def binning(df, column, bins):
    # histogram will give the boundaries of bins
    _, boundaries = np.histogram(df[column], bins=bins)
    # converting actual values into respective bins
    return df[column].apply(lambda y: bin_b(y, column, boundaries))


transactions_data = pd.DataFrame()
for column, bins in zip(
    ["age", "height_cm", "weight_kg", "value_eur"], [10, 10, 10, 20]
):
    transactions_data[column] = binning(players, column, bins)

for column in [
    "nationality",
    "club_name",
    "work_rate",
    "body_type",
]:
    transactions_data[column] = players[column].copy()

transactions_data["skill_moves"] = players["skill_moves"].apply(
    lambda x: "=%d" % x
)


transactions_data


# %%
class FPNode(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = defaultdict(FPNode)

        if parent is not None:
            parent.children[item] = self

    # Returns the top-down sequence of items from self to (but not including) the root node
    def itempath_from_root(self):
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path


# %%
def setup_fptree(df, min_support):
    num_itemsets = len(df.index)  # number of itemsets in the database

    itemsets = df.values

    # support of each individual item
    item_support = np.array(
        np.sum(itemsets, axis=0) / float(num_itemsets)
    )
    item_support = item_support.reshape(-1)

    items = np.nonzero(item_support >= min_support)[0]

    # Define ordering on items for inserting into FPTree
    indices = item_support[items].argsort()
    rank = {item: i for i, item in enumerate(items[indices])}

    # Building tree by inserting itemsets in sorted order
    # Heuristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = FPTree(rank)
    for i in range(num_itemsets):
        nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)

    return tree, rank


def generate_itemsets(generator, num_itemsets, colname_map):
    itemsets = []
    supports = []
    for sup, iset in generator:
        itemsets.append(frozenset(iset))
        supports.append(sup / num_itemsets)

    res_df = pd.DataFrame({"support": supports, "itemsets": itemsets})

    if colname_map is not None:
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([colname_map[i] for i in x])
        )

    return res_df


class FPTree(object):
    def __init__(self, rank=None):
        self.root = FPNode(None)
        self.nodes = defaultdict(list)
        self.cond_items = []
        self.rank = rank

    # generate conditional FP Tree
    def conditional_tree(self, cond_item, minsup):
        # Find all path from root node to nodes for item
        branches = []
        count = defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        # Define new ordering or deep trees may have combinatorially explosion
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        # Create conditional tree
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted(
                [i for i in branch if i in rank],
                key=rank.get,
                reverse=True,
            )
            cond_tree.insert_itemset(
                branch, self.nodes[cond_item][idx].count
            )
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

    # insert new itemset to tree
    def insert_itemset(self, itemset, count=1):
        self.root.count += count

        if len(itemset) == 0:
            return

        # Follow existing path in tree as long as possible
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if (
                len(self.nodes[i]) > 1
                or len(self.nodes[i][0].children) > 1
            ):
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print(
            "\r%d itemset(s) from tree conditioned on items (%s)"
            % (count, cond_items),
            end="\n",
        )


class FPNode(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = defaultdict(FPNode)

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path


# %%
import itertools
import math


# Get frequent itemsets from a one-hot encoded DataFrame
def fpgrowth(df, min_support=0.5, use_colnames=False, max_len=None):
    colname_map = None
    if use_colnames:
        colname_map = {
            idx: item for idx, item in enumerate(df.columns)
        }

    tree, _ = setup_fptree(df, min_support)
    minsup = math.ceil(
        min_support * len(df.index)
    )  # min support as count
    generator = fpg_step(tree, minsup, colname_map, max_len)

    return generate_itemsets(generator, len(df.index), colname_map)


# Generate frequent itemsets from FP Tree
def fpg_step(tree, minsup, colnames, max_len):
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min(
                    [tree.nodes[i][0].count for i in itemset]
                )
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (
        not max_len or max_len > len(tree.cond_items)
    ):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(
                cond_tree, minsup, colnames, max_len
            ):
                yield sup, iset


# %% [markdown]
# ## Association Rule Mining

# %%
from itertools import combinations
import numpy as np
import pandas as pd


def generate_rules(df, min_confidence=0.8):
    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, sA, __: sA,
        "consequent support": lambda _, __, sC: sC,
        "support": lambda sAC, _, __: sAC,
        "confidence": lambda sAC, sA, _: sAC / sA,
    }

    columns_ordered = [
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
    ]

    # get dict of {frequent itemset} -> support
    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # iterate over all frequent itemsets
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        # to find all possible combinations
        for idx in range(len(k) - 1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                sA = frequent_items_dict[antecedent]
                sC = frequent_items_dict[consequent]

                score = metric_dict["confidence"](sAC, sA, sC)
                if score >= min_confidence:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(
            columns=["antecedents", "consequents"] + columns_ordered
        )

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"],
        )

        sAC = rule_supports[0]
        sA = rule_supports[1]
        sC = rule_supports[2]
        for m in columns_ordered:
            df_res[m] = metric_dict[m](sAC, sA, sC)

        return df_res


# %%
encoded = pd.get_dummies(transactions_data)
encoded

# %%
pd.DataFrame(encoded.columns, columns=["column name"])

# %%
freq_itemsets = fpgrowth(encoded, min_support=0.1)
freq_itemsets

# %%
generate_rules(freq_itemsets, min_confidence=0.6)


# %% [markdown]
# ## Dynamic Itemset Counting


# %% [markdown]
# Algorithm: https://www2.cs.uregina.ca/~dbd/cs831/notes/itemsets/DIC.html
#
# SS = Æ // solid square (frequent)
#
# SC = Æ // solid circle (infrequent)
#
# DS = Æ // dashed square (suspected frequent)
#
# DC = { all 1-itemsets } // dashed circle (suspected infrequent)
#
# while (DS != 0) or (DC != 0) do begin
#
# read M transactions from database into T
#
# forall transactions t ÎTdo begin
#
# //increment the respective counters of the itemsets marked with dash
#
# for each itemset c in DS or DC do begin
#
# if ( c Î t ) then
#
# c.counter++ ;
#
# for each itemset c in DC
#
# if ( c.counter ³ threshold ) then
#
# move c from DC to DS ;
#
# if ( any immediate superset sc of c has all of its subsets in SS or DS ) then
#
# add a new itemset sc in DC ;
#
# end
#
# for each itemset c in DS
#
# if ( c has been counted through all transactions ) then
#
# move it into SS ;
#
# for each itemset c in DC
#
# if ( c has been counted through all transactions ) then
#
# move it into SC ;
#
# end
#
# end
#
# Answer = { c Î SS } ;


# %%
def subset_generator(S):
    a = itertools.combinations(S, len(S) - 1)
    for i in a:
        yield set(i)


def superset_generator(S, unique_itemset):
    a = set()
    for i in unique_itemset:
        if i.intersection(S) == set():
            a = i.union(S)
            yield a
            a = set()


# %%
def DIC(transactions, M=100, min_support=0.5):
    transactions = transactions.values
    abs_min_support = int(min_support * len(transactions))

    if len(transactions) % M != 0:
        print(len(transactions), M)
        return

    max_windows_possible = len(transactions) // M
    solid_box = {}  # frequent
    solid_circle = {}  # infrequent
    dashed_box = {}  # suspected frequent
    dashed_circle = {}  # suspected infrequent
    windows_counted = {}

    # unique items in database
    unique_items = [{e} for e in range(len(transactions[0]))]
    for i in unique_items:
        dashed_circle[tuple(i)] = 0
        windows_counted[tuple(i)] = 0

    # print(dashed_circle)

    l = 0  # current transaction
    while len(dashed_box) != 0 or len(dashed_circle) != 0:
        for t in transactions[l : l + M]:
            # indices where the items occur / index of true values
            t = set(t.nonzero()[0])
            # print(t)

            # increment the respective counters of the itemsets marked with dash
            for c in dashed_box:
                # if c.issubset(t):
                if t.issuperset(c):
                    dashed_box[c] += 1

            # increment the respective counters of the itemsets marked with dash
            for c in dashed_circle:
                # if c.issubset(t):
                if t.issuperset(c):
                    dashed_circle[c] += 1

            # since deleting while iterating will throw error
            # delete after the loop is finished
            keys = list(dashed_circle.keys())
            for c in keys:
                if dashed_circle[c] > abs_min_support:
                    # move c from DC to DS
                    dashed_box[c] = dashed_circle[c]
                    del dashed_circle[c]
                    # if any immediate superset sc of c has all of its subsets in SS or DS
                    supersets = superset_generator(c, unique_items)
                    for sc in supersets:
                        sc = tuple(sorted(sc))
                        subsets = subset_generator(sc)
                        flag = True
                        for ss in subsets:
                            ss = tuple(sorted(ss))
                            if (
                                ss not in solid_box
                                and ss not in dashed_box
                            ):
                                flag = False
                                break

                        # if all subsets are frequent start counting the superset
                        if flag:
                            dashed_circle[sc] = 0
                            windows_counted[sc] = 0
                            # add a new itemset sc in DC

            for c in list(dashed_box.keys()):
                # if c has been counted through all transactions
                if windows_counted[c] == max_windows_possible:
                    # move it to solid box
                    solid_box[c] = dashed_box[c]
                    del dashed_box[c]

            for c in list(dashed_circle.keys()):
                # if c has been counted through all transactions
                if windows_counted[c] == max_windows_possible:
                    # move it to solid circle
                    solid_circle[c] = dashed_circle[c]
                    del dashed_circle[c]

        for c in dashed_box:
            windows_counted[c] += 1

        for c in dashed_circle:
            windows_counted[c] += 1

        l = (l + M) % transactions.shape[0]
        # print(dashed_circle)
        # print(dashed_box)

    return solid_box


# %%
# dropping last few transactions to get whole number to match window size
freq_itemsets_dic = DIC(encoded[:18000], M=2000, min_support=0.2)
freq_itemsets_dic = pd.DataFrame(
    freq_itemsets_dic.items(), columns=["itemsets", "support"]
)
freq_itemsets_dic

# %% [markdown]
# ## Association rules generation
#
# This step will be the same as for FP-Growth

# %%
rules_dic = generate_rules(freq_itemsets_dic, min_confidence=0.8)
rules_dic

# %% [markdown]
# # 09/12/2023 ASBD Endsem

# %%
for col in ["antecedents", "consequents"]:
    rules_dic[col] = [
        tuple([encoded.columns[i] for i in itemset])
        for itemset in rules_dic[col]
    ]

rules_dic


# %% [markdown]
# ## Noise Removal
#
# Replace noisy data / outliers with median and quartile edges


# %%
def noise_removal(df):
    df = df.copy()
    q1, q2, q3 = df.describe()[["25%", "50%", "75%"]]
    IQR = q3 - q1
    upper_bound = q3 + 1.5 * IQR
    lower_bound = q1 - 1.5 * IQR

    count = 0
    for i in range(df.shape[0]):
        if df.iloc[i] == np.nan:
            df.iloc[i] = q2  # replace null values with median
            count += 1
        elif df.iloc[i] > upper_bound:
            df.iloc[i] = upper_bound
            count += 1
        elif df.iloc[i] < lower_bound:
            df.iloc[i] = lower_bound
            count += 1

    print("%d noisy points replace" % count)
    return df


players.wage_eur = noise_removal(players.wage_eur)
players.physic = noise_removal(players.physic)
players.defending = noise_removal(players.defending)

# %% [markdown]
# ## New plots
#
# These plots are done for midsem:
#
# - Bar chart
# - Pie chart
# - Histogram
# - Box plot
# - Violin plot
# - Spider chart
# - Scatter plot
#
# Preprocessing:
#
# - Data cleaning
# - replacing null values
# - removing duplicate records,
# - dropping useless columns like profile picture
# - data transformation
# - data selection
#
# are done in midsem itself
#
# For end sem:
#
# - Noise removal for some columns
#
# Some new plots:
#
# - Density plot
# - Rug plot
# - Hexbin plot
# - Q-Q plot
# - Heatmap
#
# ## Density and Rug plots

# %%
sns.kdeplot(players.age)
sns.rugplot(players.age)
plt.title("Age distribution using Density plot")
plt.show()

# %% [markdown]
# ### Rug plot with hexbin plot

# %%
players.plot(
    kind="hexbin",
    x="height_cm",
    y="weight_kg",
    gridsize=8,
    cmap="twilight",
)
sns.rugplot(players, x="height_cm", y="weight_kg")
plt.show()

# %% [markdown]
# - The highlighted color shows the number of points inside the colored hexagon
# - Each hexagon is basically a bin hence the name hexbin plot

# %% [markdown]
# ### Heatmap of correlation between different skills

# %%
cols = [
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
]
corr_skills = players[cols].corr(numeric_only=True)

sns.heatmap(corr_skills, annot=True, linewidth=1)
plt.show()

# %% [markdown]
# - There is a high positive correleation between dribbling and passing
# - There is a high negative correlation between defending and pace, defending and shooting
# - There is very low corellation between dribbing and physic, pace and physic

# %% [markdown]
# ### Q-Q plot

# %%
# Comparing wage with exponential distribution
_ = stats.probplot(players.wage_eur, plot=plt, dist=stats.expon())

# %%
# Comparing wage with normal distribution
_ = stats.probplot(players.wage_eur, plot=plt, dist=stats.norm())

# %%
# Comparing age with normal distribution
_ = stats.probplot(players.age, plot=plt, dist=stats.norm())

# %% [markdown]
# Age of the players mostly follows normal distribution

# %% [markdown]
# ### Plot showing joined dates of different players in FC Barcelona

# %%
top_players = players[
    players.club_name == "FC Barcelona"
].sort_values(by="overall")[:10]
dates = [datetime.strptime(d, "%Y-%m-%d") for d in top_players.joined]

# Choose some nice levels
levels = ([-5, 5, -3, 3, -1, 1] * (len(dates) // 6 + 1))[: len(dates)]

# Create figure and plot a stem plot with the date
_, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")

plt.vlines(dates, 0, levels, color="tab:red")  # The vertical stems.
plt.plot(
    dates, np.zeros_like(dates), "-o", color="k", markerfacecolor="w"
)  # Baseline and markers on it.

# annotate lines
for d, l, r in zip(dates, levels, top_players.short_name):
    plt.annotate(
        r,
        xy=(d, l),
        xytext=(-3, np.sign(l) * 3),
        textcoords="offset points",
        horizontalalignment="right",
        verticalalignment="bottom" if l > 0 else "top",
    )

# format x-axis with 4-month intervals
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

ax.yaxis.set_visible(False)
ax.spines[:].set_visible(False)
ax.spines[["bottom"]].set_visible(True)

plt.title("Timeline of Player's joined dates")
plt.show()


# %% [markdown]
# ## Algorithm 1: Apriori


# %%
# generating candidates for next level
def join(Lk):
    i = 0
    while i < len(Lk):
        skip = 1
        *ifirst, ilast = Lk[i]
        tails = [ilast]

        for j in range(i + 1, len(Lk)):
            *jfirst, jlast = Lk[j]

            if ifirst == jfirst:
                tails.append(jlast)
                skip += 1
            else:
                break

        it_first = tuple(ifirst)
        for a, b in combinations(tails, 2):
            yield it_first + (a,) + (b,)

        i += skip


# remove the candidate if any of its immediate subset is infrequent
def prune(Lk, Ck1):
    for i in Ck1:
        for j in range(len(i) - 2):
            removed = i[:j] + i[j + 1 :]

            if removed not in Lk:
                break

        else:
            yield i


# get support for all itemsets in Ck using boolean repr. of transaction database
def support(transactions, Ck):
    Cks = [0] * len(Ck)
    for i, c in enumerate(Ck):
        Cks[i] = transactions[:, c].all(axis=1).sum()

    return Cks


# give boolean transactions
def apriori(transactions, min_support=0.6):
    abs_min_support = min_support * len(transactions)
    print("min_support:", abs_min_support)

    columns = list(range(transactions.shape[1]))
    Ck = [(c,) for c in columns]
    # print(Ck)
    Cks = support(transactions, Ck)

    Lk = []  # itemsets in Lk
    Lks = []  # support for itemsets in Lk
    for c in range(len(Ck)):
        if Cks[c] >= abs_min_support:
            Lk.append(Ck[c])
            Lks.append(Cks[c])

    k = 1
    C = {}  # candidates (Ck) at each level
    L = {}  # Frequent itemsets (Lk) at each level

    while len(Lk) != 0:
        C[k] = (Ck, Cks)
        L[k] = (Lk, Lks)
        k += 1

        Ck = tuple(prune(Lk, join(Lk)))
        Cks = support(transactions, Ck)
        # print(len(Ck), len(Cks))

        Lk = []
        Lks = []
        # apply min_support and convert Ck to Lk
        for c in range(len(Ck)):
            if Cks[c] >= abs_min_support:
                Lk.append(Ck[c])
                Lks.append(Cks[c])

        # print(len(Lk), len(Lks))

    return C, L


# convert frequent itemsets to pandas DataFrame
def table(itemsets):
    C = []
    for level, items in itemsets.items():
        for item in zip(*items):
            C.append((level, *item))

    columns = ["level", "itemsets", "support"]
    return pd.DataFrame(C, columns=columns)


# %%
freq_c_apriori, freq_itemsets_apriori = apriori(
    encoded.values, min_support=0.1
)
frq_apr = table(freq_itemsets_apriori)

frq_apr["itemsets_decoded"] = [
    tuple([encoded.columns[i] for i in itemset])
    for itemset in frq_apr["itemsets"]
]
frq_apr

# %% [markdown]
# - most players (16564) have value (in eur) between 0 and 5275000
# - frequent itemset (31, 905, 913): 5541 represents that 5541 players have
#     - value_eur between 0 and 5275000
#     - work_rate as medium/medium
#     - with body type normal

# %% [markdown]
# ### Generating Association Rules
#
# This step is same for FPGrowth, DIC and Apriori

# %%
apriori_rules = generate_rules(frq_apr, min_confidence=0.8)
apriori_rules

# %% [markdown]
# - rule (892) -> (905) conf: 1.000000 mean that a player with skill_moves=1 has a work_rate of Medium/Medium always
# - rule (892) -> (905, 31) 0.920643 mean that a player with skill_moves=1 most likely has
#     - work_rate of Medium/Medium
#     - value between 0 and 5275000

# %% [markdown]
# ### Association Rules decoded

# %%
for col in ["antecedents", "consequents"]:
    apriori_rules[col] = [
        tuple([encoded.columns[i] for i in itemset])
        for itemset in apriori_rules[col]
    ]

apriori_rules


# %% [markdown]
# ## Algorithm 2: MFI (Maximal Frequent Itemset) Mining


# %%
# filter freq, infrequent itemsets using minimum support
def filter_pincer(transactions, ck, min_support):
    lk = []
    sk = []

    for c in ck:
        # if len(transactions_index_vertical(unique_items, c)) >= min_support:
        sup = transactions[:, c].all(axis=1).sum()
        if sup >= min_support:
            lk.append(c)
        else:
            sk.append(c)

    return lk, sk


# generate mfcs for the next step
def mfcs_gen(mfcs: set[tuple], sk: list[tuple]):
    for s in sk:
        for m in list(mfcs):
            ms = set(m)
            # if s.issubset(ms):
            if ms.issuperset(s):
                mfcs.remove(m)
                for e in s:
                    mse = set(m)  # - {e}
                    mse.remove(e)
                    flag = True
                    for ms in mfcs:
                        if mse.issubset(ms):
                            flag = False
                            break

                    if flag:
                        mfcs.add(tuple(sorted(mse)))
    # return mfcs


# If a candidate is a subset of any item in MFS remove it
def mfs_prune(mfs, ck):
    for c in ck:
        cs = set(c)
        for m in mfs:
            if cs.issubset(m):
                ck.remove(c)
                break


# If a candidate is not a subset of any item in MFCS remove it
def mfcs_prune(mfcs, ck1):
    for c in list(ck1):
        flag = True
        for m in mfcs:
            m = set(m)
            # if c.issubset(m):
            if m.issuperset(c):
                flag = False
                break

        if flag:
            ck1.remove(c)


# transactions in boolean representation
def pincer_search(transactions, min_support=0.5):
    min_support = int(min_support * len(transactions))

    mfcs = set()
    mfcs.add(tuple(range(transactions.shape[1])))

    mfs = set()

    ck = [(c,) for c in range(transactions.shape[1])]
    k = 1
    sk = [-1]  # infrequent itemsets

    while len(ck) != 0 and len(sk) != 0:
        print(k)
        # filter freq, infrequent itemsets from ck
        lk, sk = filter_pincer(transactions, ck, min_support)
        freq_mfcs, _ = filter_pincer(transactions, mfcs, min_support)
        mfs = mfs.union(freq_mfcs)
        if len(sk) != 0:
            mfcs_gen(mfcs, sk)

        print("level:", k)
        # print('ck:', ck)
        # print('lk:', lk)
        # print('sk:', sk)
        # print('freq_mfcs:', freq_mfcs)
        print("mfcs:", mfcs)
        # print('mfs:', mfs)
        print()

        mfs_prune(mfs, ck)
        ck = list(
            prune(lk, join(lk))
        )  # same join & prune step from apriori
        mfcs_prune(mfcs, ck)
        k += 1

    return pd.DataFrame(
        zip(mfs, support(transactions, mfs)),
        columns=["MFS", "support"],
    )


# %% [markdown]
# - Both frequent and infrequent itemset at each level is used in the algorithm
# - uses both top-down and bottom-up approach

# %%
mfs = pincer_search(encoded.values, min_support=0.1)
mfs

# %% [markdown]
# ### Decoding MFS items

# %%
mfs["mfs_decoded"] = [
    tuple([encoded.columns[i] for i in itemset])
    for itemset in mfs["MFS"]
]
mfs

# %% [markdown]
# - Maximal Frequent Itemset (13, 894): 2047 mean that all non empty subsets of (13, 894) {(13), (894), (13, 894)} are frequent with atleast a support of 2047
#     - 13 => height_cm between 175 and 180
#     - 894 => skill_moves = 3
#
# - Maximal Frequent Itemset (24, 31, 909): 2123 mean that all non empty subsets of (24, 31, 909)
#     {(24), (31), (909), (24, 31), (24, 909), (31, 909), (24, 31, 909)} are frequent with atleast a support of 2123
#     - 24 => weight_kg between 68 and 74
#     - 31 => value_eur between 0 and 5275000
#     - 909 => body_type is Lean

# %%
print("no. of Frequent Itemsets:", len(frq_apr))
print("no. of MFI's for the same min_support:", len(mfs))


# %% [markdown]
# ## Algorithm 4: Naive Bayes Classifier
#
# Taking player's position as class label, and other selected attributes as training data, perform Naive Bayes Classification


# %%
def performance_metrics(y_true, y_pred, labels=None):
    print("accuracy:", metrics.accuracy_score(y_true, y_pred))
    precision = metrics.precision_score(
        y_true, y_pred, average=None, zero_division=np.nan
    )
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    print(
        pd.DataFrame(
            [precision, recall, f1_score],
            columns=labels,
            index=["precision", "recall", "f1_score"],
        )
    )
    print("\nconfusion matrix:")

    return pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred),
        index=labels,
        columns=labels,
    )


# %% [markdown]
# ### Data Selection

# %%
attributes = [
    "physic",
    "shooting",
    "dribbling",
    "passing",
    "defending",
    "pace",
    "power_stamina",
    "skill_curve",
    "skill_long_passing",
]
training_data = players[attributes].copy()
training_data

# %% [markdown]
# ### Data Transformation

# %%
# grouping positions to get class labels
class_labels = {
    "GK": ["GK"],
    "FOR": ["ST", "LW", "RW", "LF", "RF", "RS", "LS", "CF"],
    "MID": [
        "CM",
        "RCM",
        "LCM",
        "CDM",
        "RDM",
        "LDM",
        "CAM",
        "LAM",
        "RAM",
        "RM",
        "LM",
    ],
    "DEF": ["CB", "RCB", "LCB", "LWB", "RWB", "LB", "RB"],
}

y = []
for t in players.team_position:
    for k, v in class_labels.items():
        if t in v:
            y.append(k)
            break
    else:
        y.append(-1)

y = pd.Series(y)

# removing SUB players
X = training_data.values[y != -1]
y = y[y != -1]

y

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

performance_metrics(
    y_test, y_pred, labels=sorted(class_labels.keys())
)

# %% [markdown]
# - 4 class classification with class labels
#     - GK (Goal Keeper)
#     - FOR (Forward player)
#     - MID (Midfield player)
#     - DEF (Defender)
# - Naive bayes classifier gives an accuracy around 72%
# - precision, recall, f1_score for goal keeper class is the highest and it is not misclassified almost all times
# - There as some mis-classification between midfield players and forward and defenders


# %% [markdown]
# ## Things done in Endsem Exam
#
# - [x] Noise / outliers Removal using 5 summary
# - [x] New plots
#    - [x] Density plot
#    - [x] Rug plot
#    - [x] Hexbin plot
#    - [x] Q-Q plot
#    - [x] Heatmap
# - [x] Data transformation: reducing players position into 4 unique items from
# 29 (used for classification)
# - [x] Apriori for Frequent itemset mining with association rules
# - [x] Pincer search for mining MFI's
# - [x] Naive Bayes for predicting player position based on other attributes
