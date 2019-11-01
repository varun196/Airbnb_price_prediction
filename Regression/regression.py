import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.linear_model import LinearRegression

data_train, data_test, labels_train, labels_test = None, None, None, None

with open("pickles/pca_train_test.pkl","rb") as f:
    lst = pickle.load(f)
    data_train, data_test, labels_train, labels_test = lst
    
    reg = LinearRegression().fit(data_train,labels_train)
    labels_pred = reg.predict(data_test)

    labels_test_np = labels_test.to_numpy()

    mse = metrics.mean_squared_error(labels_test_np, labels_pred)
    print("MSE",mse)

    data = None
    with open("pickles/preprocessed_data.pkl","rb") as f:
        data = pickle.load(f)
    labels = data['log_price']
    mean_label = labels.mean()
    print("Meand deviation: ", mse/mean_label * 100)