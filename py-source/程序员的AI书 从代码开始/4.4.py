from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit

if __name__ == '__main__':
    loaded_data = datasets.load_boston()
    feature = loaded_data['feature_names']
    X = loaded_data.data
    y = loaded_data.target
    mode1 = LinearRegression()
    best_model = mode1
    best_test_mse = 100
    cv = ShuffleSplit(n_splits=3, test_size=.1, random_state=0)
    for train, test in cv.split(X):
        mode1.fit(X[train], y[train])
        train_pred = mode1.predict(X[train])
        train_mse = mean_squared_error(y[train], train_pred)
        test_pred = mode1.predict(X[test])
        test_mse = mean_squared_error(y[test], test_pred)
        print('train mse:'+str(train_mse) + 'test mse:' + str(test_mse))
        if test_mse < best_test_mse:
            best_test_mse = test_mse
            best_model= mode1
    print('lr best mse score: ' + str(best_test_mse))