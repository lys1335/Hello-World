class LinearRegression(object):
    def __init__(self, eta=0.01, iterations=10):
        self.lr = eta
        self.iterations = iterations 
        self.w = 0.0
        self.bias = 0.0
    def cost_function(self, X, Y, weight, bias):
        n = len(X)
        total_error = 0.0
        for i in range(n): 
            total_error += (Y[i]-(weight*X[i]+bias))**2 
        return total_error/n 

    def update_weights(self, X, Y, weight, bias, learning_rate): 
        dw = 0 
        db = 0 
        n = len(X) 


        for i in range(n): 
            dw += -2*X[i]*(Y[i]-(weight*X[i]+bias))
            db += -2*(Y[i]-(weight*X[i]+bias)) 

        weight -= (dw/n)*learning_rate 
        bias -= (db/n)*learning_rate 

        return weight, bias 


    def fit(self, X, Y): 
        cost_history = []
        for i in range(self.iterations): 
            self.w, self.bias = self.update_weights(X, Y, self.w, self.bias, self.lr)



            #计算误差, 用于观彩和监控训练过程 
            cost = self.cost_function(X, Y, self.w, self.bias) 
            cost_history.append(cost) 

            if i % 10 == 0: 
                print("iter={:d} weight={:.2f} bias={:4f} cost={:.2}".format(i, self.w, self.bias, cost)) 

        return self.w, self.bias, cost_history 


    def predict(self, x):
        x=(x+100)/200 
        return self.w*x+self.bias 


x=[1, 2, 3, 10, 20, 50, 100, -2, -10, -100, -5, -20]
y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,0.0, 0.0]

model = LinearRegression(0.01, 500) 

x=[(k+100)/200 for k in x]

model.fit(x, y)

test_x=[90, 80, 81, 82, 75, 40, 32, 15, 5, 1, -1, -15, -20, -22, -33, -45, -60, -90]

for i in range(len(test_x)): 
    print('input {} => prediet: {}'.format(test_x[i], model.predict(test_x[i])))