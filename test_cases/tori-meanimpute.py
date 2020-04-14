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
# parser.add_argument("prob_missing",help="Probability of a point being missing",type=float)
parser.parse_args()
args = parser.parse_args()


#Torus Parameters
R1,r1 = (8,1)
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

gpu_list = get_available_gpus()
for gpu in gpu_list:
	print("Detected GPU: ", gpu )
	
#ids for the points
ids = [i for i in range(len(data))]
losses = []

#print loss every 5% 
do_print = args.epochs*10 // 100



probs =[0.3,0.0]
# probs = [0,0.002,0.004,0.006,0.008,0.010,0.012,0.014,0.016,0.018,0.02]
performance = []
amount_missing = []



for p in probs:

	model = tf.keras.Sequential([
		keras.layers.Dense(20,activation = 'relu', input_shape = (3,), 
			kernel_regularizer =tf.keras.regularizers.l2(l2reg)),
		keras.layers.Dense(5,activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(l2reg)),
		keras.layers.Dense(1, activation = 'sigmoid')
		])

	tf.keras.backend.clear_session()
	opt = tf.keras.optimizers.Adam()
	bce = tf.keras.losses.BinaryCrossentropy()


	#insert nans as missing data
	mframe = MissingData(data,labels)

	# create the observable dataset and mask:
	mframe.MCAR_Mask(p) #0.04 

	#shuffle observable data and mask
	obs_data, mask,labels = mframe.Shuffle()

	print("percentage of datapoints with missing values ",mframe.percentage_missing(),"%")
	print("percentage of empty points ",mframe.percentage_empty(),"%")
	per = mframe.percentage_missing()
	amount_missing.append(per)

	#mids = positions of points that helpave missing values, cids = positions of points that are complete
	mids,cids = mframe.Generate_ids()


	#initialise the unobservable values
	X_start = mframe.Initialise_Nans()

	for i in tqdm(range(args.epochs)):
		
		#sample batch and labels
		batch_ids = np.random.choice(ids,args.batch_size,replace=False)
		batch = X_start[batch_ids]
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
	model_per = 100*np.mean(val_preds==val_labels)
	performance.append(model_per)


	print("Percentage accuracy on validation set = ",model_per)



plt.figure(figsize=(14,6))
xaxes=np.arange(len(performance))
rects = plt.bar(xaxes,performance,align='center',alpha=0.5,color='g')
plt.xticks(xaxes,probs)
plt.xlabel(r"$P(x_i = NAN)$")
plt.ylabel("%% Validation Accuracy")
plt.title(r"$(R_1,R_2) = (%i,%i)$, $(r_1,r_2)=(%i,%i)$, $(B,T) = (%i,%i)$"%(R1
	,R2,r1,r2,args.batch_size,args.epochs))
plt.grid("on")
ax2 = plt.gca()

labels = ["%.3f%%"%(i*100) for i in amount_missing]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 0.5, label,
            ha='center', va='bottom')
ax2.set_facecolor('#D9E6E8')
plt.show()

