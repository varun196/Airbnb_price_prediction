from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import pandas as pd
import pickle

data = None

with open("../pickles/preprocessed_data_nn.pkl","rb") as f:
    data = pickle.load(f)

X = data.drop('log_price', 1)
y = data['log_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

NN_model = Sequential()

# The Hidden Layers :
NN_model.add(Dense(32, kernel_initializer='normal',input_dim = data.shape[1]-1, activation='relu'))
NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(32, kernel_initializer='normal',activation='linear'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
hist = NN_model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
#NN_model.summary()

checkpoint_name = 'checkpoints/asd-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

NN_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split = 0.2, callbacks=callbacks_list)

# plot loss during training
plt.title('Loss')
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='val')
plt.legend()