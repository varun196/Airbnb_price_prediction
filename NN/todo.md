- [x] Train-test split + plot + overfitting

- [x] K-fold cv:
```
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        # Compile model
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    # evaluate model with standardized dataset
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, Y, cv=kfold)
```
- [x] "Deeper" network
- [x] "Wider" network
- [ ] Derived feature columns:   
https://www.tensorflow.org/tutorials/estimator/linear#derived_feature_columns  

- [ ] "Feature vector" 