#import random #该种引用会报错，如下请使用 from random import random
from random import random
#下边有用到exp函数，需要import math.
import math

#手工创建两组数据分别用于训练与测试
dataset = [[2.7810836, 4.550537003, 0],
[3.396561688, 4.400293529, 0],
[1.38807019, 1.850220317, 0],
[3.06407232, 3.005305973, 0],
[7.627531214, 2.759262235, 1],
[5.332441248, 2.088626775, 1],
[6.922596716, 1.77106367, 1]]
test_data = [[1.465489372, 2.362125076, 0],
[8.675418651,-0.242068655, 1],
[7.673756466, 3.508563011, 1]]

#定义输入和输出的个数
n_inputs = 2
n_outputs = 2

#创建一个initialize_network函数来实现定义神经网络
def initialize_network_1(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(hidden_layer)
    network.append(output_layer)
    return network

#增加深度，在原有的单层隐藏层上再加一层。
def initialize_network_2(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer1 = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    hidden_layer2 = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_hidden)]
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(hidden_layer1)
    network.append(hidden_layer2)
    network.append(output_layer)
    return network

#每个神经元的网络输入
def net_input(weights, inputs):
    total_input = weights[-1]
    for i in range(len(weights)-1):
        total_input += weights[i] * inputs[i]
    return total_input

#激活函数activation
def activation(total_input):
    return 1.0/(1.0 + math.exp(-total_input))

#定义前向传播的实现
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        outputs = []
        for neuron in layer:
            total_input = net_input(neuron['weights'], inputs)
            neuron['output'] = activation(total_input)
            outputs.append(neuron['output'])
        inputs = outputs
    return inputs

#定义cost_function函数
def cost_function(expected,outputs):
    n = len(expected)
    total_error = 0.0
    for i in range (n):
        total_error += (expected[i] - outputs[i])**2
    return total_error

#sigmoid激活函数的导数实现
def transfer_derivative(output):
    return output * (1.0 - output)

#反向传播实现
def backward_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        
        if i == len(network) - 1:
            for j in range(len(layer)):
                neuron = layer[j]
                error = -2 * (expected[j] - neuron['output'])
                errors.append (error)
        else:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
            
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

#更新权重
def update_weights(netvork, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] -= learning_rate * neuron['delta']

#模型训练
def train_network(network, training_data, learning_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in training_data:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += cost_function(expected, outputs)
            backward_propagate(network, expected)
            update_weights(network, row, learning_rate)
    print('>epoch: %d, learning rate: %.3f, error: %.3f' % (epoch, learning_rate, sum_error))

#把概率最大的分类选出来
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

#运行其代码。
network = initialize_network_1(n_inputs,1, n_outputs)
#最终确认结果三组中有一组与预期不符。对应有两种方法。
#方法一，可增加神经网络的宽度，即上面1变2。
#network = initialize_network_1(n_inputs,2, n_outputs)
train_network(network, training_data = dataset, learning_rate = 0.5, n_epoch=20, n_outputs = n_outputs)
#方法二，增加深度，在原有的单层隐藏层上再加一层。涉及initialize_network的改动。并提高epoch次数到2000，可提高准确率近100%。
#network = initialize_network_2(n_inputs,1, n_outputs)
#train_network(network, training_data = dataset, learning_rate = 0.5, n_epoch=2000, n_outputs = n_outputs)

for row in test_data:
    result = predict(network, row)
    print('expected: %d, predicted: %d\n' % (row[-1], result))
