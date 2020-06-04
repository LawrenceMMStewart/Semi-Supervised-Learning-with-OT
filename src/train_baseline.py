"""
File: train_baseline
Description: This file trains a MLP on the wine dataset, with varied amounts of
labels available in order to establish a baseline for the model.
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from datasets.dataload import *
import os
import argparse 

#seed the RNG 
np.random.seed(123)
tf.random.set_seed(123)

#args = number of labels to train on
parser = argparse.ArgumentParser(description = "Training Arguements")
parser.add_argument("dataset",help="Options = wine,")
parser.add_argument("device",
    help="options = [GPU:x,CPU:0]")
parser.add_argument("n_labels",
	help = "Number of labels to train on",
	type = int)
parser.add_argument("batch_size",
	help="batch size for training [64,128,256]",
	type =int)
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
if args.device == "CPU:0":
    dev = "/"+args.device
else:
    gid = int(args.device[-1])
    tf.config.experimental.set_visible_devices(gpus[gid], 'GPU')
    dev = tf.config.experimental.list_logical_devices('GPU')[0]


with tf.device(dev):

	dname = args.dataset
	n_labels = args.n_labels
	batch_size = args.batch_size

	#save losses to tensorboard
	run_tag = "n"+str(n_labels)+"-b"+str(batch_size)
	logdir = os.path.join("src","logs",dname,"baseline",run_tag)
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

	if dname == "wine":
		train,test,train_y,test_y = load_wine()
	if dname =="diabetes":
		train,test,train_y,test_y = load_diab()
	if dname =="housing":
		train,test,train_y,test_y = load_housing()


	#only use an user-chosen amount of data for training
	train = train[:n_labels]
	train_y = train_y[:n_labels]

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

	model.compile(optimizer="adam",loss=loss_fun,
		metrics=[tf.keras.metrics.RootMeanSquaredError()])

	epochs=25000

	training_history= model.fit(train, train_y,epochs=epochs,verbose=0,
		batch_size = batch_size,
		validation_data=(test,test_y),
		callbacks=[tensorboard_callback])

	print("train set performance: \n",
		model.evaluate(x=train,y=train_y))
	print("validation set performance: \n",
		model.evaluate(x=test,y=test_y))

	#save model
	save_path = os.path.join("src","models",dname,
		"baseline",run_tag)
	model.save(save_path)

