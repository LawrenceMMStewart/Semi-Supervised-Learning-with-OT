import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



scaler = MinMaxScaler()  
path = "winequality-white.csv"
data = pd.read_csv(path,sep=';')

desc = data.describe()
print(desc.T)

fig = sns.heatmap(data.corr()
,annot=True,linewidths=.5,center=0,cmap="YlGnBu")
# plt.show()

#find any nan values:
print("Missing values? \n",data.isna().any())

"""
According to the correlation matrix, we can drop the following variables:
pH, free sulfur dioxide, residual sugar (looking at their correlation
to the wine quality.)
"""
keys_to_drop = ['pH','free sulfur dioxide','residual sugar']
data_cut = data.drop(columns=keys_to_drop)


X = data_cut.drop(columns='quality')
Y = data_cut['quality']
scaler.fit(X)

train_x,test_x,train_y,test_y =train_test_split(X,Y,
	random_state = 0, stratify = Y,shuffle=True,
	train_size=4000)

train  = scaler.transform(train_x)
test   = scaler.transform(test_x)


print(train.shape)
print(test.shape)




