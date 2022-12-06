from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

#声明两列特征数据(weather、temp)和标签数据(play), 共14组数据
weather=['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast' , 'Overcast' , 'Rainy']
temp=['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool','Mild', 'Mild', 'Mild', 'Hot', 'Mild']
play=['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

#将字符串数据通过label encoding转成数字如果特征天气对应的值可能有overcast、rainy、sunny, 购通过label encoding 转换后分别对应0,1,2.scikit-learn里面的LabelEncoder库提供了这种方法
le= preprocessing.LabelEncoder()
wheather_encoded=le.fit_transform(weather)
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

#转换后的特征和标签分别为
#wheather_encoded:[22011102212001]
#temp_encoded:[11120002022212]
#label:[001110101111101
#通过 pandas的concat 方法将两列特征合并
df1= pd.DataFrame(wheather_encoded,columns =['wheather'])
df2= pd.DataFrame(temp_encoded, columns= ['temp'])
result =pd.concat([df1, df2], axis=1, sort=False)

#合并后的特征为[(2, 1), (2, 1), (0, 1), 1, 2), (1, 0), 1, 0), (0, 0), (2,212(2, 0), 1, 2), (2, 2), (0, 2), (0, 1), (1, 211春生成朴素贝叶斯分类模型, 并将数据代入模型中进行训练
model = GaussianNB()
trainx = np.array(result)
model.fit(trainx, label)

#用生成的模型预测天气为overcast、温度为mild时的结果
predicted=model.predict([[0,2]])  #0:0vercast, 2:Mild
print("Predicted Value:",predicted)
