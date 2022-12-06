class Perceptron(object):
    def __init__(self, eta=0.01, iterations=10):
        self.lr = eta
        self.iterations = iterations 
        self.w = 0.0 
        self.bias = 0.0


    def fit(self, X, Y):
        for _ in range(self.iterations):
            for i in range(len(X)):
                x=X[i] 
                y=Y[i]
                update = self.lr * (y - self.predict(x)) 
                self.w += update * x 
                self.bias += update


    def net_input(self, x):
        return self.w * x + self.bias


    def predict(self, x):
        return 1.0 if self.net_input(x)> 0.0 else 0.0



x=[1, 2, 3, 10, 20, -2, -10, -100, -5,-20] 
y=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

model = Perceptron(0.01, 10) 
model.fit(x, y)

test_x=[30, 40, -20, -60]
for i in range(len(test_x)):
    print('input {} => predict: {}'.format(test_x[i], model.predict(test_x[i])))


print(model.w) 
print(model.bias)
