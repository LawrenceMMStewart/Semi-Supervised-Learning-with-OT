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

pred = model(test)



amplitudes = [1e-3,5e-3,1e-2,5e-2,1e-1]
mean_val_perturbs = [] #mean change across validation set
max_val_perturbs = [] #maximum change for predictions in validation set
min_val_perturbs = [] # minimum change for predictions in validation set


for stddev in amplitudes:
	noise = tf.random.normal(test.shape,mean=0.0,stddev = stddev)
	aug_pred = model(test+noise)
	mse = loss_fun(aug_pred,pred)
	diff = tf.reduce_mean(mse)

	#record the mean change (and max and min)
	mean_val_perturbs.append(diff.numpy())
	max_val_perturbs.append(tf.reduce_max(mse).numpy())
	min_val_perturbs.append(tf.reduce_min(mse).numpy())


print("mean change", mean_val_perturbs)
print("max change", max_val_perturbs)
print("min change", min_val_perturbs)



fig, ax = plt.subplots()
barWidth = 0.25

#positions to plot bars
r1 = np.arange(len(mean_val_perturbs))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.bar(r1,mean_val_perturbs,label = "Mean change in prediction - validation set",
	width=barWidth, edgecolor='white')
plt.bar(r2,max_val_perturbs,label = "Max change in prediction - validation set",
	width=barWidth, edgecolor='white')
plt.bar(r3,min_val_perturbs,label = "Min change in prediction - validation set",
	width=barWidth, edgecolor='white')
opacity = 0.6
plt.legend()
plt.grid("on",axis='y')
plt.ylabel("MSE Validation Set")
plt.xlabel(r"$\sigma$")
plt.xticks([r + barWidth for r in range(len(r1))],
	[str(a) for a in amplitudes])
plt.yscale("log")
ax.set_facecolor('#D9E6E8')
plt.show()
