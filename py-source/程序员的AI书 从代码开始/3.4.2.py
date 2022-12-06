from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing

def createModel_1():
    model = Sequential()
    model.add(Dense(32, input_shape=(13,), activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
#改进, 增加层数
def createModel_2():
    model = Sequential()
    model.add(Dense(32, input_shape=(13,), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',  optimizer='adam')
    return model

(x_train,y_train), (x_test,y_test) = boston_housing.load_data()
model = createModel_1()
model.fit(x_train,y_train, batch_size=8, epochs=100)

print("--------------------------")
print(model.metrics_names)
print(model.evaluate(x_test, y_test))
print("==========================")
print(x_test)
print("==========================x_test")
print(y_test)
print("==========================y_test")

for i in range(10):
    print("OOOOOOOOOOOO")
    print(x_test.shape)
    print(y_test.shape)
    y_pred = model.predict(x_test[i])
    print("predict:{},target: {}".format(y_pred[0][0], y_test[i]))
