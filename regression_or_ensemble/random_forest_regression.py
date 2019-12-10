def randForest(data_train, data_test, labels_train, labels_test):
  rf = RandomForestRegressor(n_estimators = 15, max_depth = 8, max_features = 'auto', min_samples_leaf = 10)
  
  rand_forest = rf.fit(data_train, labels_train)
  labels_pred, train_mse, test_mse, r2_Score = results(rand_forest, data_train, data_test, labels_train, labels_test) 

  print("R2 score : ", r2_Score)

  df=pd.DataFrame({'x': range(len(labels_test)), 'test': labels_test, 'predicted': labels_pred})
  plt.plot( 'x', 'test', data=df)
  plt.plot( 'x', 'predicted', data=df)
  plt.show()

  print("Train error: ",train_mse)
  print("Test error: ",test_mse)

  return train_mse,test_mse,r2_Score
  
# Run Random Forest Regression

rf_test_list, rf_train_list, rf_r2_score = [], [], []

# Perform 10 fold cross validation
kf, index = KFold(n_splits=10), 1
train_error_rf, test_error_rf, r2_score_avg_rf = 0, 0, 0

for train_index, test_index in kf.split(X):
  print("Round: ",str(index))
  X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]
 
  print()

  train_mse_rf,test_mse_rf, r2_score_rf = randForest(X_train, X_test, y_train, y_test)
  rf_test_list.append(test_mse_rf)
  rf_train_list.append(train_mse_rf)
  rf_r2_score.append(r2_score_rf)

  print()

  train_error_rf += train_mse_rf
  test_error_rf += test_mse_rf
  r2_score_avg_rf += r2_score_rf
  index += 1

print()

print ("Random Forest")
print("Train MSEs")
print(rf_train_list)
print("Test MSEs")
print(rf_test_list)
print("R2 scores")
print(rf_r2_score)

print()

print ("Random Forest Results")
print("Train MSE after 10-fold CV: ", str(train_error_rf/10))
print("Test MSE after 10-fold CV: ", str(test_error_rf/10))
print("R2 score after 10-fold CV: ", str(r2_score_avg_rf/10))
