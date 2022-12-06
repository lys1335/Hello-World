from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.vis_utils import plot_model

model = Sequential([
    Dense(4, input_shape=(2,)),
    Activation('sigmoid'),
    Dense(1),
    Activation('sigmoid'),
])

# 如果报错pip install pydot及graphviz错误，则需要去网站下载安装graphviz，并设置环境变量，再重启。即可。
plot_model(model, to_file='training_model.png', show_shapes=True)
