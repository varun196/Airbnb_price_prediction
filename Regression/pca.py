import pickle

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model, metrics, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import Imputer, MinMaxScaler, MultiLabelBinarizer
from sklearn.tree import DecisionTreeRegressor

data = None
with open("../pickles/preprocessed_data.pkl","rb") as f:
    data = pickle.load(f)

labels = data['log_price']
data = data.drop('log_price', 1)

# Normalize data
data = (data - data.min())/(data.max() - data.min())

# split data to train, test
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.2, random_state=0)

pca = PCA(0.85,whiten=True)  # 95% PCA component fitting
pca.fit(data)
data_train = pd.DataFrame(pca.transform(data_train))
data_test = pd.DataFrame(pca.transform(data_test))

# Save
with open("../pickles/pca_train_test.pkl","wb") as f:
    lst = [data_train, data_test, labels_train, labels_test]
    pickle.dump(lst,f)
