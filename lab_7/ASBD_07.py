# %%
from itertools import combinations
import pandas as pd
import time


# %%
def table(itemsets, vertical=False):
    C = []
    for level, items in itemsets.items():
        for item in zip(*items):
            C.append((level, *item))

    columns = ["Level", "Itemset", "Support"]
    if vertical:
        columns.append("Transactions")
    return pd.DataFrame(C, columns=columns)


# %%
def read_transactions(file_name):
    with open("../datasets/" + file_name) as file:
        for line in file:
            yield list(map(int, line.strip().split(" ")))


# %%
def convert_vertical(transactions):
    vertical_transactions = {}

    for t in range(len(transactions)):
        for i in transactions[t]:
            if i not in vertical_transactions:
                vertical_transactions[i] = [t]
            else:
                vertical_transactions[i].append(t)

    return vertical_transactions


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


def prune(Lk, Ck1):
    for i in Ck1:
        for j in range(len(i) - 2):
            removed = i[:j] + i[j + 1 :]

            if removed not in Lk:
                break

        else:
            yield i


# counting support for apriori
def support(transactions, Ck):
    Cks = [0] * len(Ck)
    for t in transactions:
        t = frozenset(t)
        for c in range(len(Ck)):
            if frozenset(Ck[c]).issubset(t):
                Cks[c] += 1
    return Cks


# counting support using vertical transaction database
def support_vertical(Lk, Lkt, Ck1):
    Cks = []
    Ckt = []
    for c in Ck1:
        c1, c2 = c[:-1], c[1:]
        for i in range(len(Lk)):
            if Lk[i] == c1:
                s1 = Lkt[i]
            elif Lk[i] == c2:
                s2 = Lkt[i]

        common = s2.intersection(s1)

        Cks.append(len(common))
        Ckt.append(common)

    return Cks, Ckt


# %%
def apriori(transactions, min_support=0.6):
    abs_min_support = min_support * len(transactions)
    print("min_support:", abs_min_support)
    unique_items = {}
    for t in transactions:
        for i in t:
            if i in unique_items:
                unique_items[i] += 1
            else:
                unique_items[i] = 1

    unique_items = sorted(unique_items.items())
    Ck, Cks = tuple(zip(*unique_items))
    Ck = tuple((c,) for c in Ck)
    Lk = []
    Lks = []

    for i in range(len(Ck)):
        if Cks[i] >= abs_min_support:
            Lk.append(Ck[i])
            Lks.append(Cks[i])

    k = 1
    C = {}
    L = {}

    while len(Lk) != 0:
        C[k] = (Ck, Cks)
        L[k] = (Lk, Lks)
        k += 1

        Ck = tuple(prune(Lk, join(Lk)))
        Cks = support(transactions, Ck)

        Lk = []
        Lks = []
        for i in range(len(Ck)):
            if Cks[i] >= abs_min_support:
                Lk.append(Ck[i])
                Lks.append(Cks[i])

    return C, L


# %% [markdown]
# # Apriori Vertical Transaction notation (ECLAT)


# %%
# Apriori algorithm using vertical transaction database
def apriori_vertical(vt, size, min_support=0.6):
    abs_min_support = min_support * size
    print("min_support:", abs_min_support)
    Ck, Ckt = tuple(zip(*sorted(vt.items())))
    Ck = tuple((c,) for c in Ck)
    Cks = tuple(len(t) for t in Ckt)
    Ckt = tuple(frozenset(t) for t in Ckt)

    Lk = []
    Lks = []
    Lkt = []
    for c in range(len(Ck)):
        if Cks[c] >= abs_min_support:
            Lk.append(Ck[c])
            Lks.append(Cks[c])
            Lkt.append(Ckt[c])

    k = 1
    C = {}
    L = {}

    while len(Lk) != 0:
        C[k] = (Ck, Cks, Ckt)
        L[k] = (Lk, Lks, Lkt)
        k += 1

        Ck = tuple(prune(Lk, join(Lk)))
        Cks, Ckt = support_vertical(Lk, Lkt, Ck)

        Lk = []
        Lks = []
        Lkt = []
        for c in range(len(Ck)):
            if Cks[c] >= abs_min_support:
                Lk.append(Ck[c])
                Lks.append(Cks[c])
                Lkt.append(Ckt[c])

    return C, L


# %% [markdown]
# # Apriori Transaction Reduction


# %%
def apriori_transaction_reduction(transactions, min_support=0.6):
    abs_min_support = min_support * len(transactions)
    tn = len(transactions)
    sot = [len(t) for t in transactions]
    print("min_support:", abs_min_support)
    unique_items = {}
    for t in transactions:
        for i in t:
            if i in unique_items:
                unique_items[i] += 1
            else:
                unique_items[i] = 1

    unique_items = sorted(unique_items.items())
    Ck, Cks = tuple(zip(*unique_items))
    Ck = tuple((c,) for c in Ck)
    Lk = []
    Lks = []

    for i in range(len(Ck)):
        if Cks[i] >= abs_min_support:
            Lk.append(Ck[i])
            Lks.append(Cks[i])

    k = 1
    C = {}
    L = {}

    while len(Lk) != 0:
        C[k] = (Ck, Cks)
        L[k] = (Lk, Lks)
        k += 1

        Ck = tuple(prune(Lk, join(Lk)))
        Cks = [0] * len(Ck)

        for i in range(tn - 1, -1, -1):
            # remove non frequent items from database
            contains = False
            ti = frozenset(transactions[i])
            for l in Lk:
                l = frozenset(l)
                if l.issubset(ti):
                    contains = True
                    break

            if not contains:
                del transactions[i]
                del sot[i]
                tn -= 1
                continue

            # delete if sot less than itemset size
            if sot[i] < k:
                del transactions[i]
                del sot[i]
                tn -= 1
                continue

            for c in range(len(Ck)):
                ci = frozenset(Ck[c])
                if ci.issubset(ti):
                    Cks[c] += 1

        Lk = []
        Lks = []
        for i in range(len(Ck)):
            if Cks[i] >= abs_min_support:
                Lk.append(Ck[i])
                Lks.append(Cks[i])

    return C, L


# %% [markdown]
# # Apriori Dynamic Hashing and Pruning


# %%
def apriori_dhp(transactions, min_support=0.6):
    ct = transactions.copy()
    abs_min_support = min_support * len(transactions)
    print("min_support:", abs_min_support)
    unique_items = {}
    for t in transactions:
        for i in t:
            if i in unique_items:
                unique_items[i] += 1
            else:
                unique_items[i] = 1

    unique_items = sorted(unique_items.items())
    Ck, Cks = tuple(zip(*unique_items))
    Ck = tuple((c,) for c in Ck)
    Lk = []
    Lks = []

    for i in range(len(Ck)):
        if Cks[i] >= abs_min_support:
            Lk.append(Ck[i])
            Lks.append(Cks[i])

    k = 1
    C = {}
    L = {}
    Ck_hash = {}

    while len(Lk) != 0:
        C[k] = (Ck, Cks)
        L[k] = (Lk, Lks)
        k += 1

        Ck_hash.clear()
        if k < 4:
            for i in range(len(ct)):
                if k == 2:
                    ct[i] = tuple((c,) for c in ct[i])
                else:
                    ct[i] = tuple(prune(Lk, join(ct[i])))
                for j in ct[i]:
                    if j in Ck_hash:
                        Ck_hash[j] += 1
                    else:
                        Ck_hash[j] = 1

            Ck = []
            Cks = []
            Lk = []
            Lks = []
            for fi, s in Ck_hash.items():
                Ck.append(fi)
                Cks.append(s)
                if s >= abs_min_support:
                    Lk.append(fi)
                    Lks.append(s)

        else:
            Ck = tuple(prune(Lk, join(Lk)))
            Cks = support(transactions, Ck)

            Lk = []
            Lks = []
            for i in range(len(Ck)):
                if Cks[i] >= abs_min_support:
                    Lk.append(Ck[i])
                    Lks.append(Cks[i])

    return C, L


# %%
toy = list(read_transactions("toy.txt"))
transactions = list(read_transactions("kosarak.dat"))
transactions_2 = list(read_transactions("T10I4D100K.dat"))
print("Database T10I4D100K.dat size:", len(transactions))
print("Database kosarak.dat size:", len(transactions_2))
times = []

# %%
candidates, freq = apriori_vertical(
    convert_vertical(toy),
    size=len(toy),
    min_support=2 / len(toy),
)
table(freq, vertical=True)

# %%
start = time.perf_counter()
candidates, freq = apriori_vertical(
    convert_vertical(transactions_2),
    size=len(transactions_2),
    min_support=0.008,
)
end = time.perf_counter()
times.append(["T10I4D100K.dat", "vertical", 0.008, end - start])
table(freq, vertical=True)

# %%
table(candidates, vertical=True)

# %%
start = time.perf_counter()
candidates, freq = apriori_vertical(
    convert_vertical(transactions),
    size=len(transactions),
    min_support=0.03,
)
end = time.perf_counter()
times.append(["kosarak.dat", "vertical", 0.03, end - start])
table(freq, vertical=True)

# %%
table(candidates, vertical=True)

# %%
transactions_copy = toy.copy()
candidates, freq = apriori_transaction_reduction(
    transactions_copy, min_support=2 / 9
)
table(freq)

# %%
transactions_copy = transactions.copy()
start = time.perf_counter()
candidates, freq = apriori_transaction_reduction(
    transactions_copy, min_support=0.03
)
end = time.perf_counter()
times.append(
    [
        "kosarak.dat",
        "reduction",
        0.03,
        end - start,
    ]
)
table(freq)

# %%
table(candidates)

# %%
candidates, freq = apriori_dhp(toy, min_support=2 / 9)
table(freq)

# %%
start = time.perf_counter()
candidates, freq = apriori_dhp(transactions_2, min_support=0.008)
end = time.perf_counter()
times.append(["T10I4D100K.dat", "DHP", 0.008, end - start])
table(freq)

# %%
table(candidates)

# %%
start = time.perf_counter()
candidates, freq = apriori_dhp(transactions_2, min_support=0.03)
end = time.perf_counter()
times.append(["T10I4D100K.dat", "DHP", 0.03, end - start])
table(freq)

# %%
table(candidates)

# %%
# # Comparison

# %%
pd.DataFrame(
    times,
    columns=["dataset", "algorithm", "min_support", "time taken"],
)
