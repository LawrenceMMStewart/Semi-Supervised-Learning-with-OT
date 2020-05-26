"""
File: augment
Description: Evaluate the performance of MLP's (trained on various levels
of noisy data) on the validation set.  
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


checkpoint_dir  = os.path.join("src","models",dname,"aug_noise")

accuracy = []
noise_amp = []
for i in range(1,6):
	mname = "1.00E-0"+"%i"%i
	mdir = os.path.join(checkpoint_dir,mname)
	model = tf.keras.models.load_model(mdir)
	pred = model(test)
	mse = tf.reduce_mean(loss_fun(pred,test_y))
	accuracy.append(mse.numpy())
	noise_amp.append(10**(-i))


print("models acc",list(zip(noise_amp,accuracy)))
fig, ax = plt.subplots()
barWidth = 0.25
r = np.arange(len(accuracy))
plt.bar(r,accuracy,edgecolor='black')
opacity = 0.6

plt.grid("on",axis='y')
plt.ylabel("MSE Validation Set")
plt.xlabel(r"$\sigma$")
plt.xticks([a for a in range(len(r))],
	[str(a) for a in noise_amp])
plt.yscale("log")
ax.set_facecolor('#D9E6E8')
plt.show()
