"""
File: tori-classifier.py 
Description: train a neural network to classifier between two noisy tori
    		(this is a test for the Impute classifier experiment to ensure
    		the classifier as a base case can succeed at the task.)

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
from utils.datasets import *
from sklearn.utils import shuffle
from utils.utils import *
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
import pickle 
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(123)


parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("batch_size",help = "|Xkl|",type = int) 
parser.add_argument("epochs",help = "Epochs of round robin imputations", type = int)

parser.parse_args()
args = parser.parse_args()


#Torus Parameters
R1,r1 = (12,1)
R2,r2 = (5,1)
n_sq = 30 
var = 3

Tori = NoisyTorus(R1,R2,r1,r2,n_sq,n_sq,var)

data = Tori.data
labels = Tori.labels 

#create validation set
n2_sq = 15
Val = NoisyTorus(R1,R2,r1,r2,n2_sq,n2_sq,var)
val_data = Val.data
val_labels = Val.labels

l2reg=1e-4

model = tf.keras.Sequential([
	keras.layers.Dense(20,activation = 'relu', input_shape = (3,), 
		kernel_regularizer =tf.keras.regularizers.l2(l2reg)),
	keras.layers.Dense(5,activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2reg)),
	keras.layers.Dense(1, activation = 'sigmoid')
	])

model.summary()
#do we need a flatten?

opt = tf.keras.optimizers.Adam()
gpu_list = get_available_gpus()
for gpu in gpu_list:
	print("Detected GPU: ", gpu )
	

ids = [i for i in range(len(data))]


bce = tf.keras.losses.BinaryCrossentropy()
losses = []

#print loss every 5% 
do_print = args.epochs*10 // 100

for i in tqdm(range(args.epochs)):
	
	#sample batch and labels
	batch_ids = np.random.choice(ids,args.batch_size,replace=False)
	batch = data[batch_ids]
	batch_labels = labels[batch_ids]
	
	with tf.GradientTape() as tape:
		pred = model(batch)
		loss = bce(pred,batch_labels)

	losses.append(loss)
	#update gradients
	gradients = tape.gradient(loss ,model.trainable_weights)
	opt.apply_gradients(zip(gradients,model.trainable_weights))

	if i % do_print ==0:
		tqdm.write("Loss = %.5f"%loss)





val_preds = model(val_data).numpy()
val_preds[val_preds>=0.5]=1
val_preds[val_preds<0.5]=0
print(val_preds)
print("Percentage accuracy on validation set = ",100*np.mean(val_preds==val_labels))

