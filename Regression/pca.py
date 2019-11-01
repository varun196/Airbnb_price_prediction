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

data = None
with open("pickles/preprocessed_data.pkl","rb") as f:
    data = pickle.load(f)

labels = data['log_price']
data = data.drop('log_price', 1)

# Normalize data
data = (data - data.min())/(data.max() - data.min())

# split data to train, test
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.2, random_state=0)

pca = PCA(0.99,whiten=True)  # 95% PCA component fitting
data_train = pd.DataFrame(pca.fit_transform(data_train))
data_test = pd.DataFrame(pca.transform(data_test))

# Save
with open("pickles/pca_train_test.pkl","wb") as f:
    lst = [data_train, data_test, labels_train, labels_test]
    pickle.dump(lst,f)
