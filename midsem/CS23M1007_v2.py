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
# # Data Preprocessing
#
# # Data Cleaning
#
# ## Filling null values with median
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
# ## Dropping Records
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
# ## Remove duplicates

# %%
players.drop_duplicates(inplace=True)

# %% [markdown]
# # Data Visualization

# %% [markdown]
# ## Pie chart

# %% [markdown]
# ### Preferred foot for players

# %%
pfoot, counts = np.unique(players.preferred_foot, return_counts=True)
plt.pie(counts, labels=pfoot, autopct="%0.2f%%")
plt.legend()
plt.show()

# %% [markdown]
# # Bar Chart

# %%
plt.figure(figsize=(10, 5))
nationality, counts = np.unique(
    players.nationality, return_counts=True
)
plt.bar(nationality[10:20], counts[10:20])
plt.show()

# %% [markdown]
# ## Histogram, Box, Violin plot

# %% [markdown]
# ### Age distribution of all players

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
# ### Height distribution of all players

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
# ### Weight distribution of all players

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
# ## Scatter plots
#
# ### Comparing Height & Weight

# %%
plt.scatter(players.height_cm, players.weight_kg)
plt.xlabel("Height in cm")
plt.ylabel("Weight in Kg")
plt.show()

# %% [markdown]
# ### Comparing potential and value

# %%
players.plot(kind="scatter", x="potential", y="value_eur")
plt.show()

# %% [markdown]
# As the potential of player increases the value (in EUR) increases expoentially

# %% [markdown]
# ## Box plot

# %% [markdown]
# ### Comparing Age between different clubs

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
# ## Spider / Radar Plot

# %% [markdown]
# ### Comparing different players


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
# # FP-Growth

# %% [markdown]
# ## Data selection


# %%
# will return the range the given data is
def bin_b(x, column, boundaries):
    for j in range(1, len(boundaries)):
        if x < boundaries[j]:
            return (
                column
                + "="
                + str(int(boundaries[j - 1]))
                + "-"
                + str(int(boundaries[j]))
            )
    return column + ">" + str(int(boundaries[-1]))


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
    "skill_moves",
    "work_rate",
    "body_type",
]:
    transactions_data[column] = players[column].apply(
        lambda x: column + "=" + str(x)
    )

transactions_data


# %%
class FPNode(object):
    def __init__(self, item, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode)

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
import collections
import numpy as np
import pandas as pd

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
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank

    # generate conditional FP Tree
    def conditional_tree(self, cond_item, minsup):
        # Find all path from root node to nodes for item
        branches = []
        count = collections.defaultdict(int)
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
        self.children = collections.defaultdict(FPNode)

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
def fpgrowth(
    df, min_support=0.5, use_colnames=False, max_len=None
):
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
# # Dynamic Itemset Counting


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
    unique_items = [
        {e} for e in range(len(transactions[0]))
    ]
    for i in unique_items:
        dashed_circle[tuple(i)] = 0
        windows_counted[tuple(i)] = 0

    # print(dashed_circle)

    l = 0  # current transaction
    while len(dashed_box) != 0 or len(dashed_circle) != 0:
        for t in transactions[l:l+M]:
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
generate_rules(freq_itemsets_dic, min_confidence=0.8)
