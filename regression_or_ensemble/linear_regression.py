def linear_reg_model(data_train, data_test, labels_train, labels_test):
  cov_matrix = data_train.cov()
  eigen_values,eigen_vectors = np.linalg.eig(cov_matrix)
  #plt.plot(range(len(eigen_values)),eigen_values,alpha=0.7, color = "green")
  #plt.xlabel("Principle Component")
  #plt.ylabel("Eigen value")
  #plt.title("Eigen Value Plot")
  #plt.show()

  pca = PCA(n_components = 85) 
  pca.fit(data_train) # fit using train and transform train and test data
  data_train = pd.DataFrame(pca.transform(data_train)) 
  data_test = pd.DataFrame(pca.transform(data_test))

  reg = LinearRegression().fit(data_train,labels_train)
  labels_pred = reg.predict(data_test)
  labels_test_np = labels_test.to_numpy()
  test_mse = metrics.mean_squared_error(labels_test_np, labels_pred)
  #print("Test MSE",test_mse)

  labels_train_pred = reg.predict(data_train)
  train_mse = metrics.mean_squared_error(labels_train.to_numpy(), labels_train_pred)
  print("r2:",metrics.r2_score(labels_test_np, labels_pred))
  #print("Train MSE",train_mse)
  #mean_label = labels.mean()
  #print("Mean deviation: ", mse/mean_label * 100)

  df=pd.DataFrame({'x': range(len(labels_test_np)), 'test': labels_test_np, 'predicted': labels_pred })
  plt.plot( 'x', 'test', data=df)
  plt.plot( 'x', 'predicted', data=df)
  plt.show()
  return train_mse,test_mse
