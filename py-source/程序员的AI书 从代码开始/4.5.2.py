import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

if __name__ == '__main__':
    # 数据文件路径
    path = 'dataset/iris.data.csv'
    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes  # 将第四列的3个结果值分类别映射成0，1，2

    X, y = np.split(data, (4,), axis=1)  # 将数据集按列拆分，前4列为X，最后一列为Y

    # 仅使用前两列特征
    X = X.iloc[:, :2]  # 所有行，前两列。此处原代码使用X[:,:2]会报索引错误，改用「iloc」

    lr = Pipeline([('sc', StandardScaler()),
                   ('poly', PolynomialFeatures(degree=3)),
                   ('clf', LogisticRegression())])
    lr.fit(X, y.values.ravel())  # y.ravel()会报错，使用y.values.ravel()
    y_hat = lr.predict(X)
    y_hat_prob = lr.predict_proba(X)
    np.set_printoptions(suppress=True)
    print('y_hat = \n', y_hat)
    print('y_hat_prob = \n', y_hat_prob)
    print('准确度: %.2f%%' % (100 * np.mean(y_hat == y.values.ravel())))
