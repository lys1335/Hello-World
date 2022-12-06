import numpy
from keras.models import Sequential
from keras.layers import Dense

numpy.random.seed(7)
# 读取糖尿病训练数据
dataset = numpy.loadtxt("dataset/pima-indians-diabetes.data.csv", delimiter=",")
# 将数据分为训练数据和标签, 前 8 列为特征
X = dataset[:, 0:8]
Y = dataset[:, 8]
# 创建模型, 这边创建3层神经网络, 分别为输入层、隐藏层、输出层
model = Sequential()
model.add(Dense(4, input_dim=8, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# 编译模型
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
# 将特征和标签放入模型中
model.fit(X, Y, epochs=10, batch_size=32)
# 衡量模型效果
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
