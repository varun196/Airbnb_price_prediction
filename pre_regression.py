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
from sklearn.model_selection import KFold
import xgboost

labels = data['log_price']
data = data.drop('log_price', 1)

# Normalize data
data = (data - data.min())/(data.max() - data.min())

print(data.shape)
cov_matrix = data.cov()
eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)
plt.plot(range(len(eigen_values)),eigen_values,alpha=0.7, color = "blue")
plt.xlabel("Principle Component")
plt.ylabel("Eigen value")
plt.title("Eigen Value Plot")
plt.show()
