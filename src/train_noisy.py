"""
File: train_noisy
Description: This file trains a MLP on the wine dataset, with varied amounts of
noise added to the points in the training dataset (no noise in validation set)
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import argparse 
from decimal import Decimal


#seed the RNG 
np.random.seed(123)
tf.random.set_seed(123)

#args = number of labels to train on
parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("dataset",help="Options = wine,",type=str)
parser.add_argument("noise_amp",help="amplitude of noise",type=float)
args = parser.parse_args()

dname = args.dataset
noise_amp  = args.noise_amp
str_noise = '%.2E' % Decimal(noise_amp)

#save losses to tensorboard
run_name = str_noise + "-"+datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("src","logs",dname,"aug_noise",run_name)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

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

#add noise to training datasets
noise = tf.random.normal(train.shape, mean=0.0, stddev = noise_amp)
train = train+noise


l2reg=1e-3
d=train.shape[1]
loss_fun = tf.keras.losses.MSE



model = tf.keras.Sequential([
	tf.keras.layers.Dense(2*d, activation ='relu',input_shape = (d,),
		kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
	tf.keras.layers.Dense(d,activation = 'relu',
		kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
	tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(l2reg))
	])


model.summary()
#create a callback that saves models weights after training

model.compile(optimizer="adam",loss=loss_fun)

training_history= model.fit(train, train_y,epochs=25000,verbose=1,
	validation_data=(test,test_y),
	callbacks=[tensorboard_callback])

print("train set performance: \n",
	model.evaluate(x=train,y=train_y))
print("validation set performance: \n",
	model.evaluate(x=test,y=test_y))

#save model
save_path = os.path.join("src","models",dname,
	"aug_noise",str_noise)
model.save(save_path)

