# %% [markdown]
# # PS- IX Data Classification

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from io import StringIO


# %%
def performance_metrics(y_true, y_pred):
    print('accuracy:', accuracy_score(y_true, y_pred))
    print('precision:', precision_score(y_true, y_pred, average=None, zero_division=np.nan))
    print('recall:', recall_score(y_true, y_pred, average=None))
    print('f1 score:', f1_score(y_true, y_pred, average=None))

    return pd.DataFrame(confusion_matrix(y_true, y_pred))


# %%
toy_data = """
Outlook,Temperature,Humidity,Windy,Play Golf
Overcast,Cool,Normal,True,Yes
Overcast,Hot,High,False,Yes
Overcast,Hot,Normal,False,Yes
Overcast,Mild,High,True,Yes
Rainy,Cool,Normal,False,Yes
Rainy,Hot,High,False,No
Rainy,Hot,High,True,No
Rainy,Mild,High,False,No
Rainy,Mild,Normal,True,Yes
Sunny,Cool,Normal,False,Yes
Sunny,Cool,Normal,True,No
Sunny,Mild,High,False,Yes
Sunny,Mild,High,True,No
Sunny,Mild,Normal,False,Yes
"""

toy_dataset = pd.read_csv(StringIO(toy_data))
toy_dataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 488} executionInfo={"elapsed": 13, "status": "ok", "timestamp": 1699874116116, "user": {"displayName": "CS23M1007 VIMALRAJ S", "userId": "01696524563088723467"}, "user_tz": -330} id="GeQDzJNt6_2i" outputId="7a66d49f-8b9b-43eb-a986-0e9e3549f9c8"
train = pd.DataFrame()
for c in toy_dataset.columns:
    train[c] = toy_dataset[c].astype("category").cat.codes

X = train[train.columns[:-1]]
y = train[train.columns[-1]]

train

# %% [markdown]
# ## Decision Tree

# %%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X.values, y.values)
plt.figure(figsize=(10, 8))
tree.plot_tree(clf, filled=True)
plt.show()

# %%
print(tree.export_text(clf))

# %%
test = pd.DataFrame(
    [
        # test record (Rainy, Cool, Normal, True)
        [1, 0, 1, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0],
    ],
    columns=X.columns,
)
test_dt = test.copy()
test_dt["predicted"] = clf.predict(test_dt.values)
test_dt["predicted Play Golf"] = test_dt.predicted.replace(
    [1, 0], ["Yes", "No"]
)
test_dt


# %%
def describe_decision_tree(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    values = clf.tree_.value

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack: list[tuple[int, int]] = [
        (0, 0)
    ]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = (
            children_left[node_id] != children_right[node_id]
        )
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        "The binary tree structure has {n} nodes and has "
        "the following tree structure:\n".format(n=n_nodes)
    )
    for i in range(n_nodes):
        if is_leaves[i]:
            print(
                "{space}node={node} is a leaf node with value={value}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    value=values[i],
                )
            )
        else:
            print(
                "{space}node={node} is a split node with value={value}: "
                "go to node {left} if X[:, {feature}] <= {threshold} "
                "else to node {right}.".format(
                    space=node_depth[i] * "\t",
                    node=i,
                    left=children_left[i],
                    feature=feature[i],
                    threshold=threshold[i],
                    right=children_right[i],
                    value=values[i],
                )
            )


describe_decision_tree(clf)

# %% [markdown]
# ## Naive Bayes

# %%
gnb = GaussianNB()
test_nb = test.copy()
test_nb["predicted"] = gnb.fit(X.values, y.values).predict(
    test_nb.values
)
test_nb["predicted play golf"] = test_nb.predicted.replace(
    [1, 0], ["Yes", "No"]
)
test_nb

# %% [markdown]
# # Nearest Neighbours Classifier

# %%
test_knn = test.copy()
neigh = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
neigh.fit(X.values, y.values)
test_knn["predicted"] = neigh.predict(
    test_knn.values
)
test_knn["predicted play golf"] = test_knn.predicted.replace(
    [1, 0], ["Yes", "No"]
)

test_knn

# %% [markdown]
# # Using Car Evaluation Dataset

# %%
# !pip install ucimlrepo

# %%
from ucimlrepo import fetch_ucirepo

# fetch dataset
car_evaluation = fetch_ucirepo(id=19)

# %%
# data (as pandas dataframes)
X1 = car_evaluation.data.features
y1 = car_evaluation.data.targets

# variable information
car_evaluation.variables

# %%
X1

# %%
y1

# %% [markdown]
# ## Encoding

# %%
X = pd.DataFrame()
encoders = []
for c in X1.columns:
    le = LabelEncoder()
    X[c] = le.fit_transform(X1[c])
    print(c, le.classes_)
    encoders.append(le)

le = LabelEncoder()
y = le.fit_transform(y1["class"])
print("class", le.classes_)
encoders.append(le)

X

# %%
y

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.3, random_state=0
)

# %% [markdown]
# ## Decision Tree

# %%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
plt.figure(figsize=(10, 8))
tree.plot_tree(clf, filled=True)
plt.show()

# %%
y_pred_dt = clf.predict(X_test)
y_pred_dt

# %%
performance_metrics(y_test, y_pred_dt)

# %%
describe_decision_tree(clf)

# %% [markdown]
# ## Naive Bayes

# %% [markdown]
# ### Gaussian Naive Bayes

# %%
gnb = GaussianNB()
y_pred_nb = gnb.fit(X_train, y_train).predict(X_test)

y_pred_nb

# %%
performance_metrics(y_test, y_pred_nb)

# %% [markdown]
# ### Categorical Naive Bayes

# %%
cnb = CategoricalNB()
y_pred_cnb = cnb.fit(X_train, y_train).predict(X_test)

y_pred_cnb

# %%
performance_metrics(y_test, y_pred_cnb)

# %% [markdown]
# # Nearest Neighbour Classifier

# %%
# k = 7 gives highest accuracy for this dataset
neigh = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
neigh.fit(X_train, y_train)
y_pred_knn = neigh.predict(X_test)

y_pred_knn

# %%
performance_metrics(y_test, y_pred_knn)

# %%
a = pd.DataFrame()

for (i, c), le in zip(enumerate(X.columns), encoders[:-1]):
    a[c] = le.inverse_transform(X_test[:, i])

le = encoders[-1]
a["y_test"] = le.inverse_transform(y_test)
a["y_pred_dt"] = le.inverse_transform(y_pred_dt)
a["y_pred_nb"] = le.inverse_transform(y_pred_nb)
a["y_pred_cnb"] = le.inverse_transform(y_pred_cnb)
a["y_pred_knn"] = le.inverse_transform(y_pred_knn)

a
