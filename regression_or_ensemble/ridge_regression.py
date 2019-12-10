def ridge_reg_model(data_train, data_test, labels_train, labels_test):
  cov_matrix = data_train.cov()
  eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)

  pca = PCA(n_components = 85) 
  pca.fit(data_train) # fit using train and transform train and test data
  data_train = pd.DataFrame(pca.transform(data_train)) 
  data_test = pd.DataFrame(pca.transform(data_test))

  reg = RidgeCV(cv = 10).fit(data_train,labels_train)
  labels_pred, train_mse, test_mse, r2_Score = results(reg, data_train, data_test, labels_train, labels_test) 

  print("R2 score : ", r2_Score)

  df=pd.DataFrame({'x': range(len(labels_test)), 'test': labels_test, 'predicted': labels_pred })
  plt.plot( 'x', 'test', data=df)
  plt.plot( 'x', 'predicted', data=df)
  plt.show()
  
  print("Train error: ",train_mse)
  print("Test error: ",test_mse)

  return train_mse,test_mse,r2_Score
  
# Run Ridge Regression

ridge_test_list, ridge_train_list, ridge_r2_score = [], [], []

# Perform 10 fold cross validation
kf, index = KFold(n_splits=10), 1
train_error_ridge, test_error_ridge, r2_score_avg_ridge = 0, 0, 0

for train_index, test_index in kf.split(X):
  print("Round: ",str(index))
  X_train, X_test, y_train, y_test = X.loc[train_index], X.loc[test_index], y.loc[train_index], y.loc[test_index]
  
  print()

  train_mse_ridge,test_mse_ridge, r2_score_ridge = ridge_reg_model(X_train, X_test, y_train, y_test)
  ridge_test_list.append(test_mse_ridge)
  ridge_train_list.append(train_mse_ridge)
  ridge_r2_score.append(r2_score_ridge)

  print()

  train_error_ridge += train_mse_ridge
  test_error_ridge += test_mse_ridge
  r2_score_avg_ridge += r2_score_ridge
  index += 1

print()

print ("Ridge Regression")
print("Train MSEs")
print(ridge_train_list)
print("Test MSEs")
print(ridge_test_list)
print("R2 scores")
print(ridge_r2_score)

print()

print ("Ridge Regression Results")
print("Train MSE after 10-fold CV: ", str(train_error_ridge/10))
print("Test MSE after 10-fold CV: ", str(test_error_ridge/10))
print("R2 score after 10-fold CV: ", str(r2_score_avg_ridge/10))
