def xgboosterFunc(data_train, data_test, labels_train, labels_test):

  xgb = xgboost.XGBRegressor(colsample_bytree=0.8, subsample=0.8, learning_rate=0.05, max_depth=3, 
                              min_child_weight=3, n_estimators=1000,gamma=0.0, silent=1)
  
  xgb.fit(data_train, labels_train)
  xgb_train_pred = xgb.predict(data_train)
  xgb_test_pred = xgb.predict(data_test)
  labels_test_np = labels_test.to_numpy()
  train_mse = metrics.mean_squared_error(labels_train.to_numpy(), xgb_train_pred)
  test_mse = metrics.mean_squared_error(labels_test.to_numpy(), xgb_test_pred)
  print("r2:",metrics.r2_score(labels_test_np, xgb_test_pred))
  df=pd.DataFrame({'x': range(len(labels_test_np)), 'test': labels_test_np, 'predicted': xgb_test_pred })
  plt.plot( 'x', 'test', data=df)
  plt.plot( 'x', 'predicted', data=df)
  plt.show()

  return train_mse, test_mse
