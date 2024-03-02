# %%
from itertools import combinations
import pandas as pd
from efficient_apriori import itemsets_from_transactions


# %%
def apriori_test(transactions, min_support=0.5):
    fi, _ = itemsets_from_transactions(
        transactions, min_support=min_support
    )
    out = []
    for level in fi:
        for itemset, sc in fi[level].items():
            out.append((level, itemset, sc))

    return pd.DataFrame(out, columns=["level", "itemset", "support"])


# %% [markdown]
# # 1 ACLOSE
#
# Test Drive ACLOSE algorithm to mine closed frequent patterns on a sample dataset of your
# choice. Test the same on a FIMI benchmark dataset which you have used for Apriori/FP-growth
# implementations.


# %%
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


def subsets(s):
    for i in range(len(s)):
        yield s[:i] + s[i + 1 :]


# %%
def transactions_index(transactions, itemset):
    itemset = set(itemset)
    for i in range(len(transactions)):
        if itemset.issubset(transactions[i]):
            yield i


def transactions_index_vertical(unique_items, itemset):
    out = set(unique_items[itemset[0]])
    for i in itemset[1:]:
        out = out.intersection(unique_items[i])
    return out


def items_common(transactions, indices):
    out = set(transactions[next(indices)])
    sc = 1
    for i in indices:
        out = out.intersection(transactions[i])
        sc += 1
    return out, sc


def support(transactions, ck):
    cks = [0] * len(ck)
    for t in transactions:
        t = set(t)
        for i in range(len(ck)):
            if set(ck[i]).issubset(t):
                cks[i] += 1

    return cks


def support_vertical(unique_items, ck):
    for c in ck:
        yield len(transactions_index_vertical(unique_items, c))


def filter_ck(ck, cks, lout, min_support):
    lk = []
    lks = []
    for i in range(len(ck)):
        if cks[i] >= min_support:
            lk.append(ck[i])
            lks.append(cks[i])
            lout[ck[i]] = cks[i]
    return lk, lks


# %%
def minimal_generators(transactions, min_support):
    unique_items = {}
    for j, t in enumerate(transactions):
        for i in t:
            if i not in unique_items:
                unique_items[i] = []
            unique_items[i].append(j)

    ck = []
    cks = []
    for c in unique_items:
        ck.append((c,))
        cks.append(len(unique_items[c]))

    lk = []
    lks = []
    lout = {}

    lk, lks = filter_ck(ck, cks, lout, min_support)

    k = 1
    while len(lk) != 0:
        k += 1
        ck = list(prune(lk, join(lk)))
        # cks = support(transactions, ck)
        cks = list(support_vertical(unique_items, ck))

        lk = []
        lks = []

        for i in range(len(ck)):
            if cks[i] >= min_support:
                flag = True
                for s in subsets(ck[i]):
                    if lout[s] == cks[i]:
                        flag = False
                        break

                if flag:
                    lk.append(ck[i])
                    lks.append(cks[i])
                    lout[ck[i]] = cks[i]

    return lout


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
        cfi.items(), columns=["closed itemset", "support"]
    )


# %%
transactions_1 = [
    ["A", "C", "D"],
    ["B", "C", "E"],
    ["A", "B", "C", "E"],
    ["B", "E"],
    ["A", "B", "C", "E"],
]

aclose(transactions_1, min_support=3 / 5)

# %%
apriori_test(transactions_1, min_support=3 / 5)

# %%
transactions = []

with open("../datasets/mushroom.dat") as file:
    for line in file:
        transactions.append(tuple(map(int, line.strip().split(" "))))

# %%
aclose(transactions, min_support=0.4)

# %%
apriori_test(transactions, min_support=0.4)


# %% [markdown]
# # PINCER SEARCH
#
# Test Drive Pincer search to mine maximal frequent patterns on a sample dataset of your
# choice. Test the same on a FIMI benchmark dataset which you have used for Apriori/FP-growth
# implementations.


# %%
# filter infrequent itemsets using minimum support
def filter_pincer(unique_items, ck, min_support):
    lk = []
    sk = []

    for c in ck:
        if (
            len(transactions_index_vertical(unique_items, c))
            >= min_support
        ):
            lk.append(c)
        else:
            sk.append(c)

    return lk, sk


def mfcs_gen(mfcs: set[tuple], sk: list[tuple]):
    for s in sk:
        for m in list(mfcs):
            ms = set(m)
            # if s.issubset(m):
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


def pincer_search(transactions, min_support=0.5):
    min_support = int(min_support * len(transactions))
    unique_items = {}
    for j, t in enumerate(transactions):
        for i in t:
            if i not in unique_items:
                unique_items[i] = []
            unique_items[i].append(j)

    mfcs = {tuple(sorted(unique_items.keys()))}
    mfs = set()

    ck = [(c,) for c in sorted(unique_items)]
    k = 1
    sk = [-1]

    while len(ck) != 0 and len(sk) != 0:
        lk, sk = filter_pincer(unique_items, ck, min_support)
        freq_mfcs, _ = filter_pincer(unique_items, mfcs, min_support)
        mfs = mfs.union(freq_mfcs)
        if len(sk) != 0:
            mfcs_gen(mfcs, sk)

        print('level:', k)
        # print('ck:', ck)
        # print('lk:', lk)
        print('sk:', sk)
        # print('freq_mfcs:', freq_mfcs)
        print('mfcs:', mfcs)
        # print('mfs:', mfs)
        print()

        mfs_prune(mfs, ck)
        ck = list(prune(lk, join(lk)))
        mfcs_prune(mfcs, ck)
        k += 1

    return pd.DataFrame(
        zip(mfs, support_vertical(unique_items, mfs)),
        columns=["MFS", "Support"],
    )


# %%
pincer_search(transactions_1, min_support=3 / 5)

# %%
pincer_search(transactions, min_support=0.4)

# %% [markdown]
# For minimum support of 0.4 in mushroom dataset
# - 565 frequent itemsets
# - 140 closed frequent itemsets
# - 41 maximal frequent itemsets
