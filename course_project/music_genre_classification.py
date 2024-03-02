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
# # MUSIC GENRE CLASSIFICATION
#
# - CS23M1007 VIMALRAJ S
# - CS23M1008 SARVESH
# - CS23M1009 OM PRAKASH
#
# https://www.kaggle.com/datasets/purumalgi/music-genre-classification/

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import combinations
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# %% [markdown]
# # Descriptive Analytics

# %%
music = pd.read_csv("./train.csv")
print("shape:", music.shape)
class_name = [
    "Acoustic/Folk_0",
    "Alt_Music_1",
    "Blues_2",
    "Bollywood_3",
    "Country_4",
    "HipHop_5",
    "Indie Alt_6",
    "Instrumental_7",
    "Metal_8",
    "Pop_9",
    "Rock_10",
]
music.describe()

# %% [markdown]
# ## DataType & No. of unique items

# %%
pd.DataFrame(
    [
        (c, music[c].dtype, len(music[c].unique()))
        for c in music.columns
    ],
    columns=["column", "dtype", "no. of unique_items"],
)

# %% [markdown]
# ## Count of Null values

# %%
null_values = music.isnull().sum()
null_values[null_values > 0]

# %% [markdown]
# ## Data Preprocessing
#
# ## Data Cleaning
#
# ### Filling null values with median / mode

# %%
music["Popularity"].fillna(
    music["Popularity"].mode()[0], inplace=True
)
music["key"].fillna(music.key.mode()[0], inplace=True)
music["instrumentalness"].fillna(
    music.instrumentalness.median(), inplace=True
)

null_values = music.isnull().sum()
null_values

# %% [markdown]
# ## Remove duplicates

# %%
music.drop_duplicates(inplace=True)

# %% [markdown]
# # Data Visualization

# %%
uniq_values, counts = np.unique(music["mode"], return_counts=True)
plt.pie(counts, labels=uniq_values, autopct="%0.2f%%")
plt.legend()
plt.show()

# %%
uniq_values, counts = np.unique(music["Class"], return_counts=True)
plt.pie(counts, labels=class_name, autopct="%0.1f%%")
# plt.legend()
plt.show()

# %%
plt.hist(music.loudness)
plt.axvline(music.loudness.mean(), label="mean", color="red")
plt.legend()
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.hist(music.energy)
plt.xlabel("Energy")
plt.axvline(music.energy.mean(), label="mean", color="red")
plt.legend()

plt.subplot(212)
plt.violinplot(music.energy, vert=False)
plt.boxplot(music.energy, vert=False, patch_artist=True)
plt.xlabel("Energy")
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.subplot(211)
music.tempo.plot(kind="hist")
plt.axvline(music.tempo.mean(), label="mean", color="red")
plt.legend()

plt.subplot(212)
music.tempo.plot(kind="box", vert=False)
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.subplot(211)
plt.axvline(music.liveness.mean(), label="mean", color="red")
music.liveness.plot(kind="hist")
plt.legend()

plt.subplot(212)
music.liveness.plot(kind="box", vert=False)
plt.show()

# %%
plt.scatter(music.energy, music.loudness)
plt.xlabel("Energy")
plt.ylabel("Loudness")
plt.show()

# %% [markdown]
# Music energy and loudness are correlated exponentially

# %%
music.acousticness.plot(kind="kde")
sns.rugplot(music.acousticness)

# %% [markdown]
# ## Correlation between different attributes

# %%
cols = [
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
]
corr_skills = music[cols].corr(numeric_only=True)

sns.heatmap(corr_skills, annot=True, linewidth=1)
plt.show()

# %% [markdown]
# ## Comparing tempo of different songs by same artist

# %%
x = music["Artist Name"].unique()[:10]
plt.figure(figsize=(20, 10))

for i, c in enumerate(x):
    plt.boxplot(
        music[music["Artist Name"] == c].tempo,
        positions=[i],
        widths=0.5,
        labels=[c],
    )

plt.show()

# %% [markdown]
# - The artists The Raincoats, Solomon Burke, Professional Murder Music all have same tempo for all their songs
# - Randy Travis has more songs in Low tempo
# - Bruno mars have more songs in high tempo
# - Red Hot Chilli Peppers have songs in low and medium tempo with few excepitional outliers

# %% [markdown]
# ## Q-Q plot

# %%
_ = stats.probplot(music.valence, plot=plt, dist=stats.norm())

# %%
music.plot(
    kind="hexbin",
    x="tempo",
    y="loudness",
    gridsize=8,
    cmap="twilight",
)
sns.rugplot(music, x="tempo", y="loudness")
plt.show()


# %% [markdown]
# ## Noise / Outlier Removal


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
        # replace outliers with upper and lower bounds
        elif df.iloc[i] > upper_bound:
            df.iloc[i] = upper_bound
            count += 1
        elif df.iloc[i] < lower_bound:
            df.iloc[i] = lower_bound
            count += 1

    print("%d noisy points replace" % count)
    return df


music.valence = noise_removal(music.valence)
music.tempo = noise_removal(music.tempo)

# %% [markdown]
# # Assocaition Rule Mining

# %%
music.columns

categorical = [
    #'Artist Name',
    "mode",
    "time_signature",
    "Class",
]

numerical = [
    "Popularity",
    "danceability",
    "energy",
    "key",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

all_cols = categorical + numerical

all_cols


# %% [markdown]
# ## Binning Numerical Attributes


# %%
# will return the range the given data is
def bin_b(x, column, boundaries):
    for j in range(1, len(boundaries)):
        if x < boundaries[j]:
            return "(%.2f,%.2f)" % (
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
for column, bins in zip(numerical, [7] * len(numerical)):
    transactions_data[column] = binning(music, column, bins)

# for column in categorical:
#    transactions_data[column] = players[column].copy()

for column in categorical:
    transactions_data[column] = music[column].apply(
        lambda x: "=%d" % x
    )


transactions_data

# %%
encoded = pd.get_dummies(transactions_data)
encoded

# %%
pd.DataFrame(encoded.columns, columns=["column name"])


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
min_support = 0.1
freq_c_apriori, freq_itemsets_apriori = apriori(
    encoded.values, min_support=min_support
)
frq_apr = table(freq_itemsets_apriori)

frq_apr["itemsets_decoded"] = [
    tuple([encoded.columns[i] for i in itemset])
    for itemset in frq_apr["itemsets"]
]
frq_apr

# %% [markdown]
# The itemset (8, 21, 29, 31, 33, 35, 38, 49, 52, 56) has support count of 1962

# %%
for c in (8, 21, 29, 31, 33, 35, 38, 49, 52, 56):
    print(c, "->", encoded.columns[c])

# %%
apriori_rules = generate_rules(frq_apr, min_confidence=0.8)
apriori_rules

# %% [markdown]
# The rule (17, 10, 21, 49) -> (33, 35, 38, 8, 52, 29, 31)	has confidence 90.27%

# %%
for c in (17, 10, 21, 49, 33, 35, 38, 8, 52, 29, 31):
    print(c, "->", encoded.columns[c])

# %%
for col in ["antecedents", "consequents"]:
    apriori_rules[col] = [
        tuple([encoded.columns[i] for i in itemset])
        for itemset in apriori_rules[col]
    ]

apriori_rules


# %% [markdown]
# ## CFI A-close


# %%
def subsets(s):
    for i in range(len(s)):
        yield s[:i] + s[i + 1 :]


def minimal_generators(transactions, min_support):
    columns = list(range(transactions.shape[1]))
    Ck = [(c,) for c in columns]
    # print(Ck)
    Cks = support(transactions, Ck)

    Lout = {}
    Lk = []  # itemsets in Lk
    Lks = []  # support for itemsets in Lk
    for c in range(len(Ck)):
        if Cks[c] >= min_support:
            Lk.append(Ck[c])
            Lks.append(Cks[c])
            Lout[Ck[c]] = Cks[c]

    k = 1
    while len(Lk) != 0:
        k += 1
        Ck = list(prune(Lk, join(Lk)))
        Cks = support(transactions, Ck)

        Lk = []
        Lks = []

        for i in range(len(Ck)):
            if Cks[i] >= min_support:
                flag = True
                for s in subsets(Ck[i]):
                    if Lout[s] == Cks[i]:
                        flag = False
                        break

                if flag:
                    Lk.append(Ck[i])
                    Lks.append(Cks[i])
                    Lout[Ck[i]] = Cks[i]

    return Lout


def transactions_index(transactions, itemset):
    common_indices = transactions[:, itemset].all(axis=1)
    return common_indices.nonzero()[0]
    # out = set(unique_items[itemset[0]])
    # for i in itemset[1:]:
    #    out = out.intersection(unique_items[i])
    # return out


def items_common(transactions, indices):
    common_items = transactions[indices].all(axis=0)
    return common_items.nonzero()[0], len(indices)
    # out = set(transactions[next(indices)])
    # sc = 1
    # for i in indices:
    #    out = out.intersection(transactions[i])
    #    sc += 1
    # return out, sc


def aclose(transactions, min_support=0.5):
    N = len(transactions)
    abs_min_support = int(min_support * N)

    mg = minimal_generators(transactions, abs_min_support)

    cfi = {}
    for itemset in mg:
        indices = transactions_index(transactions, itemset)
        common_items, sc = items_common(transactions, indices)
        common = tuple(sorted(common_items))
        cfi[common] = sc

    return pd.DataFrame(
        cfi.items(), columns=["closed_itemset", "support"]
    )


# %%
cfi = aclose(encoded.values, min_support=min_support)
cfi

# %% [markdown]
# CFI (3, 29, 33): 4794 represent that
# all subsets (3), (29), (33), (3, 29), (3, 33), (29, 33) have the same support

# %%
for c in (3, 29, 33):
    print(c, "->", encoded.columns[c])

# %% [markdown]
# ## MFI Pincer Search


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
        # filter freq, infrequent itemsets from ck
        lk, sk = filter_pincer(transactions, ck, min_support)
        freq_mfcs, _ = filter_pincer(transactions, mfcs, min_support)
        mfs = mfs.union(freq_mfcs)
        if len(sk) != 0:
            mfcs_gen(mfcs, sk)

        # print("level:", k)
        # print('ck:', ck)
        # print('lk:', lk)
        # print('sk:', sk)
        # print('freq_mfcs:', freq_mfcs)
        # print("mfcs:", mfcs)
        # print('mfs:', mfs)
        # print()

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


# %%
mfi = pincer_search(encoded.values, min_support=min_support)
mfi

# %%
mfi["mfs_decoded"] = [
    tuple([encoded.columns[i] for i in itemset])
    for itemset in mfi["MFS"]
]

mfi

# %% [markdown]
# The MFI (3, 8, 21, 29, 31, 33, 35, 38, 49): 1907 represents all subsets have atleast 1907 support count

# %%
for c in (3, 8, 21, 29, 31, 33, 35, 38, 49):
    print(c, "->", encoded.columns[c])

# %% [markdown]
# ## Comparing FI, CFI, MFI

# %%
print(
    "no. of Frequent Itemsets for min_support=%.2f: %d"
    % (min_support, len(frq_apr))
)
print("no. of CFI's for the same min_support:", len(cfi))
print("no. of MFI's for the same min_support:", len(mfi))

# %% [markdown]
# # PYSPARK
#
# ## Using pyspark to run FPGrowth

# %%
from pyspark.context import SparkContext
from pyspark.mllib.stat import Statistics
from pyspark.mllib.fpm import FPGrowth
from pyspark.mllib.fpm import FPGrowthModel

sc = SparkContext(appName="FPGrowth")

# %%
transactions_spark = []
for i, row in enumerate(encoded.values):
    transactions_spark.append(np.nonzero(row)[0])

# transactions_spark

# %%
with open("./transactions_spark.dat", "w") as file:
    for line in transactions_spark:
        file.write(" ".join(map(str, line)) + "\n")

# %%
data = sc.textFile("./transactions_spark.dat")
transactions = data.map(lambda line: line.strip().split(" "))

model = FPGrowth.train(
    transactions, minSupport=min_support, numPartitions=10
)
spark_result = model.freqItemsets().collect()

print("no. of frequent itemsets:", len(spark_result))

# %%
res = pd.DataFrame(spark_result)
print("min_support:", min_support)
res

# %% [markdown]
# Itemset [49, 52, 29, 8, 33] has support count 10445

# %%
for c in [49, 52, 29, 33]:
    print(c, "->", encoded.columns[c])

# %%
sc.stop()


# %% [markdown]
# # Classification


# %%
def performance_metrics(y_true, y_pred, labels=None):
    labels = class_name
    print("accuracy:", metrics.accuracy_score(y_true, y_pred))
    precision = metrics.precision_score(
        y_true, y_pred, average=None, zero_division=np.nan
    )
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(
        y_true, y_pred, average=None, zero_division=np.nan
    )

    sensitivity = []
    specificity = []
    FPR = []
    FNR = []

    for cl in range(len(class_name)):
        tp = tn = fp = fn = 0
        for i, pr in zip(y_true, y_pred):
            if pr == cl:
                if i == cl:
                    tp += 1
                else:
                    fp += 1
            else:
                if i == cl:
                    fn += 1
                else:
                    tn += 1

        sensitivity.append(tp / (tp + fn))
        specificity.append(tn / (tn + fp))
        FPR.append(fp / (fp + tn))
        FNR.append(fn / (fn + tp))

    print(
        pd.DataFrame(
            [
                precision,
                recall,
                f1_score,
                sensitivity,
                specificity,
                FPR,
                FNR,
            ],
            columns=labels,
            index=[
                "precision",
                "recall",
                "f1_score",
                "sensitivity(TPR)",
                "specificity(TNR)",
                "FPR",
                "FNR",
            ],
        )
    )
    print(metrics.classification_report(y_true, y_pred))
    print("\nconfusion matrix:")

    return pd.DataFrame(
        metrics.confusion_matrix(y_true, y_pred),
        index=labels,
        columns=labels,
    )


# %%
X_cols = all_cols[:]
X_cols.remove("Class")

X_train, X_test, y_train, y_test = train_test_split(
    music[X_cols].values, music.Class.values, test_size=0.2
)

# %% [markdown]
# ## Naive Bayes

# %%
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

performance_metrics(y_test, y_pred)

# %% [markdown]
# Accuracy around 36% for Naive Bayes, 32% KNN, 30% Decision Tree

# %% [markdown]
# ## K-Nearest Neighbour

# %%
neigh = KNeighborsClassifier(n_neighbors=80, metric="euclidean")
neigh.fit(X_train, y_train)
y_pred_knn = neigh.predict(X_test)

performance_metrics(y_test, y_pred_knn)

# %% [markdown]
# Naive Bayes performs better than KNN classifier for this dataset

# %% [markdown]
# ## Decision Tree

# %%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

y_pred_dt = clf.predict(X_test)
performance_metrics(y_test, y_pred_dt)
