"""
File: augment
Description: This file loads a pretrained MLP for the baseline task
and experiments with adding random noise to the points, finding the maximum
amplitude of noise that we can add without affecting the decision of the network.
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import argparse 
#seed the RNG 
np.random.seed(123)
tf.random.set_seed(123)

#args = number of labels to train on
parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("dataset",help="Options = wine,",type = str)
parser.add_argument("model_name",help = "name of model",type = str)
args = parser.parse_args()

dname = args.dataset


#initialisations for dataset
scaler = MinMaxScaler()  
if dname =="wine":
	path = os.path.join("datasets","wine","winequality-white.csv")
data = pd.read_csv(path,sep=';')
X = data.drop(columns='quality')
Y = data['quality']
#fit the scaler to X 
scaler.fit(X)

#split into train and test sets
train_x,test_x,train_y,test_y = train_test_split(X,Y,
	random_state = 0, stratify = Y,shuffle=True,
	train_size=4000)

train  = scaler.transform(train_x)
test   = scaler.transform(test_x)
train_y  = pd.DataFrame.to_numpy(train_y).reshape(-1,1).astype(np.float32)
test_y  = pd.DataFrame.to_numpy(test_y).reshape(-1,1).astype(np.float32)




l2reg=1e-3
d=train.shape[1]
loss_fun = tf.keras.losses.MSE


checkpoint_dir  = os.path.join("src","models",dname,"baseline",args.model_name)

model = tf.keras.models.load_model(checkpoint_dir)
model.summary()
print(model.weights)