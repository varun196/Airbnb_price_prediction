# split data to train, test
#data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.2, random_state=0)

linear_test_list = []
xgboost_test_list = []
linear_train_list = []
xgboost_train_list = []

# perfrom 10 fold cross validation:
index = 1
train_error_lin = 0
test_error_lin = 0 
train_error_xgboost = 0
test_error_xgboost = 0 
kf = KFold(n_splits=10)
for train_index, test_index in kf.split(data):
  print("Round: ",str(index))
  X_train, X_test = data.loc[train_index], data.loc[test_index]
  y_train, y_test = labels.loc[train_index], labels.loc[test_index]

  train_mse_lin,test_mse_lin = linear_reg_model(X_train, X_test, y_train, y_test)
  print("Train error: ",train_mse_lin)
  print("Test error: ",test_mse_lin)
  linear_test_list.append(test_mse_lin)
  linear_train_list.append(train_mse_lin)

  train_mse_xgboost,test_mse_xgboost = xgboosterFunc(X_train, X_test, y_train, y_test)
  print("Train error: ",train_mse_xgboost)
  print("Test error: ",test_mse_xgboost)
  xgboost_test_list.append(test_mse_xgboost)
  xgboost_train_list.append(train_mse_xgboost)

  train_error_lin+=train_mse_lin
  test_error_lin+=test_mse_lin
  train_error_xgboost+=train_mse_xgboost
  test_error_xgboost+=test_mse_xgboost
  index+=1

print ("Traditional Linear Regression")
print("Train mse after 10-fold CV: ", str(train_error_lin/10))
print("Test mse after 10-fold CV: ", str(test_error_lin/10))

print ("xgboost")
print("Train mse after 10-fold CV: ", str(train_error_xgboost/10))
print("Test mse after 10-fold CV: ", str(test_error_xgboost/10))

print("linear_test_list")
print(linear_test_list)
print("linear_train_list")
print(linear_train_list)
print("xgboost_test_list")
print(xgboost_test_list)
print("xgboost_train_list")
print(xgboost_train_list)
