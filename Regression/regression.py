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
import matplotlib.pyplot as plt

data_train, data_test, labels_train, labels_test = None, None, None, None

with open("../pickles/pca_train_test.pkl","rb") as f:
    lst = pickle.load(f)
    data_train, data_test, labels_train, labels_test = lst
    
reg = LinearRegression().fit(data_train,labels_train)
labels_pred = reg.predict(data_test)

labels_test_np = labels_test.to_numpy()

mse = metrics.mean_squared_error(labels_test_np, labels_pred)
print("MSE",mse)

data = None
with open("../pickles/preprocessed_data.pkl","rb") as f:
    data = pickle.load(f)
labels = data['log_price']
mean_label = labels.mean()
print("Mean deviation: ", mse/mean_label * 100)

# for ind in range(len(labels_test_np)):
#     print(labels_test_np[ind],labels_pred[ind])