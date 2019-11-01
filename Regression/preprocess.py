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

# Read data set
data = pd.read_csv("../data/train.csv") 

columns = ['id', 'city', 'zipcode', 'neighbourhood', 'description', 'name', 'thumbnail_url']
data.drop(columns, inplace = True, axis=1)

# Review date should not affect price.
cols = ['first_review', 'last_review', 'host_since']
data.drop(cols, inplace = True, axis=1)

# Transform columns
data['cleaning_fee'] = data['cleaning_fee'].astype(int)
data['instant_bookable'] = data['instant_bookable'].map({'f': 0, 't': 1})
data['host_has_profile_pic'] = data['host_has_profile_pic'].map({'f': 0, 't': 1})
data['host_identity_verified'] = data['host_identity_verified'].map({'f': 0, 't': 1})

# Remove % sign from host_response_rate
def process_host_resp(s):
    if isinstance(s, str):
        return float(s[:-1])
data['host_response_rate'] = data['host_response_rate'].apply(process_host_resp)

# One hot encoding - property type, room type, amenities, bed type, cancellations, 
categorical=['property_type','room_type','bed_type','cancellation_policy']
data = pd.get_dummies(data, columns=categorical)


data['amenities'] = data['amenities'].apply(lambda s: s.replace('"', "").replace('{', "").replace('}', ""))
data['amenities'] = data['amenities'].apply(lambda s: s.split(","))

mlb = MultiLabelBinarizer()
data = data.join(pd.DataFrame(mlb.fit_transform(data.pop('amenities')),
                          columns=mlb.classes_,
                          index=data.index))

# Fill missing data with medians
data = data.fillna(data.median())

with open("pickles/preprocessed_data.pkl","wb") as f:
    pickle.dump(data,f)

