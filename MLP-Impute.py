"""
File: MLP_toy_examples.py 
Description: This file implements the data imputation via sinkorn using a batch approach 
for sklearn toy datasets. It explores the effect of altering the sinkhorn divergance's
ground cost and regularisation parameter.

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from utils.utils import *
from utils.sinkhorn import *
import tensorflow as tf
from tqdm import tqdm
import pickle 
from tensorflow import keras
np.random.seed(123)


parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("batch_size",help = "|Xkl|",type = int) 
parser.add_argument("epsilon", help = "Sinkhorn Divergance Regularisation Parameter",type = float)
parser.add_argument("exponent" , help = "Exponent of euclidean distance (1 or 2)", type = float)
parser.parse_args()
args = parser.parse_args()

name = str(args.batch_size)+'-'+str(args.epsilon)+'-'+str(args.exponent)

n_samples =  1000 #1000
data,_ = datasets.make_s_curve(n_samples=n_samples)


#insert nans as missing data
mframe = MissingData(data)

# create the observable dataset and mask:
mframe.MCAR_Mask(0.1) #0.15 for the other 2 

#shuffle observable data and mask
obs_data, mask = mframe.Shuffle()

print("percentage of datapoints with missing values ",mframe.percentage_missing(),"%")
print("percentage of empty points ",mframe.percentage_empty(),"%")
per = mframe.percentage_missing()
#mids = positions of points that have missing values, cids = positions of points that are complete
mids,cids = mframe.Generate_Labels()



d=mframe.m
Imputers = [] 
# for a test create a MLP for co-ordinate 1
#do we need a flatten?
for j in range(d):
	Imputers.append(tf.keras.Sequential([
	keras.layers.Dense(2*(d-1), activation ='relu',input_shape = (d-1,)),
	keras.layers.Dense(d-1,activation = 'relu'),
	keras.layers.Dense(1)
	]))

batchsize = args.batch_size
assert batch_size % 2 == 0 ; "batchsize corresponds to the size of Xkl and hence must be even sized"

#potentially use the tf.compat.v1.train.Adam version if this has problems
opt = tf.keras.optimizers.Adam()
gpu_list = get_available_gpus()

#initialise the unobservable values
X_start = mframe.Initialise_Nans()
 


#minisise sinkhorn distance for each stochastic batch  
X = X_start.copy() #X_0 in the algorithm
epochs = 100
train_epochs = 1000  #K in paper
# indicies = [i for i in range(n_samples)]

#clears old sessions
tf.keras.backend.clear_session()

for t in tqdm(range(epochs),desc = "Iteration"):
	for j in tqdm(range(len(Imputers)),desc = "Round Robin"):
		for k in tqdm(range(train_epochs),desc = "Training Imputer"):
			#Xk has no missing values on dim j, Xl has only missing values on dim 
			kl_indicies = mframe.getbatch_jids(batch_size,j,replace=False)
			#create Xkl and its mask 

			with tf.GradientTape() as tape:

				#create Xkl and its mask
				Xkl = tf.constant(X[kl_indicies]) #MAYBE THIS SHOULD BE A VARIABLE
				msk = mask[kl_indicies]


				#indicies of dimensions that are not j
				notj = [i for i in range(d) if i!=j]

				#predict dimension j 
				pred_j = Imputers[j](Xkl[:,notj])

				
				
				#calculate loss
				loss = sinkhorn_sq_batch(Xkl_fill,p=2,niter=10,div=True,epsilon=0.1)

				#gradients with respect to network parameters
				gradients = tape.gradient(loss , )
				opt.apply_gradients






# #stochastic batch gradient descent w.r.t sinkhorn divergance
# for t in tqdm(range(epochs),desc = "Epoch"):
# 	#sample the two batches whos concatenation is Xkl
# 	kl_indicies = np.random.choice(indicies,batch_size,replace=True)
# 	Xkl = tf.Variable(X[kl_indicies])
# 	msk = mask[kl_indicies] #mask of Xkl
# 	loss_fun = lambda: sinkhorn_sq_batch(Xkl,p=args.exponent,niter=10,div=True,
# 		epsilon=args.epsilon) 

# 	#compute gradient
# 	grads_and_vars = opt.compute_gradients(loss_fun)

# 	#mask the gradient (i.e. so we are calculating w.r.t to X_impute)
# 	mskgrads_and_vars=[(g*(1-msk),v) for g,v in grads_and_vars]
# 	#apply gradient step
# 	opt.apply_gradients(mskgrads_and_vars)
# 	#update X:
# 	X[kl_indicies] = Xkl.numpy()

# #calculate and save the imputed data
# imputed_data = np.nan_to_num(obs_data*mask) + X*(1-mask)



