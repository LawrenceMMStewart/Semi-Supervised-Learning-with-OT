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
from tensorflow import keras
import pickle 
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(123)


parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("batch_size",help = "|Xkl|",type = int) 
parser.add_argument("epsilon", help = "Sinkhorn Divergance Regularisation Parameter",type = float)
parser.add_argument("exponent" , help = "Exponent of euclidean distance (1 or 2)", type = float)
parser.add_argument("epochs",help = "Epochs of round robin imputations", type = int)
parser.add_argument("train_epochs",help = "training iterations ", type = int)
parser.parse_args()
args = parser.parse_args()

name = str(args.batch_size)+'-'+str(args.epsilon)+'-'+str(args.exponent)+'-'+str(args.epochs)+'-'+str(args.train_epochs)

n_samples =  1000 #1000
data,_ = datasets.make_s_curve(n_samples=n_samples)


#insert nans as missing data
mframe = MissingData(data)

# create the observable dataset and mask:
mframe.MCAR_Mask(0.04) #0.15 for the other 2 

#shuffle observable data and mask
obs_data, mask = mframe.Shuffle()

print("percentage of datapoints with missing values ",mframe.percentage_missing(),"%")
print("percentage of empty points ",mframe.percentage_empty(),"%")
per = mframe.percentage_missing()
#mids = positions of points that have missing values, cids = positions of points that are complete
mids,cids = mframe.Generate_ids()


l2reg=1e-4
d=mframe.m
Imputers = [] 
# for a test create a MLP for co-ordinate 1
#do we need a flatten?
for j in range(d):
	Imputers.append(tf.keras.Sequential([
	keras.layers.Dense(2*(d-1), activation ='relu',input_shape = (d-1,),
		kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
	keras.layers.Dense(d-1,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
	keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(l2reg))
	]))

batch_size = args.batch_size
assert batch_size % 2 == 0 ; "batchsize corresponds to the size of Xkl and hence must be even sized"

#potentially use the tf.compat.v1.train.Adam version if this has problems
opt = tf.keras.optimizers.Adam()
gpu_list = get_available_gpus()

#initialise the unobservable values
X_start = mframe.Initialise_Nans()
 

#minisise sinkhorn distance for each stochastic batch  
Xt = X_start.copy() #X_0, X_1, ... X_T in the algorithm
epochs = args.epochs # overall training epochs
train_epochs = args.train_epochs  #K i.e number of stochastic gradient descent steps
n_average = 3   # number of sinkhorn divergances to average over 


for t in tqdm(range(epochs),desc = "Iteration"):
	for j in tqdm(range(len(Imputers)),desc = "Round Robin"):

		#create a mask just for dimension j (i.e. we consider the other missing values as present)
		maskj = np.ones(mask.shape)
		maskj[:,j] = mask[:,j]

		#indicies of dimensions that are not j
		notj = tf.constant([i for i in range(d) if i!=j])

		for k in tqdm(range(train_epochs),desc = "Training Imputer"):
			
			#Create X_hat by imputing Xt  
			X_no_j = tf.gather(Xt,notj,axis=1).numpy()
			X_pred = Imputers[j](X_no_j)
			X_hat = Xt*maskj + X_pred*(1-maskj)

			#gradient descent w.r.t MLP weights over average 
			#sinkhorn divergances of batches
			with tf.GradientTape() as tape:
				mean_loss = tf.constant(0.)
				for mean_epochs in range(n_average):
					#Xk has no missing values on dim j, Xl has only missing values on dim 
					kl_indicies = mframe.getbatch_jids(batch_size,j,replace=True)
					#Sample the batch as a variable (maybe a constant works)
					Xkl = tf.constant(X_hat.numpy()[kl_indicies]) # maybe this needs to be a tensorflow variable 
					
					#create the mask msk and the mask for axis j
					msk = mask[kl_indicies]
					#mask for column j 
					mskj = np.ones(msk.shape)
					mskj[:,j] = msk[:,j]

					#retrieve Xkl without the j'th dimension
					Xkl_no_j = tf.gather(Xkl,notj,axis=1)				

					#predict dimension j 
					pred_j = Imputers[j](Xkl_no_j)
			                       
					#impute the data on dimension j
					Xkl_imputed = Xkl*mskj + (1-mskj)*pred_j 	
					
					#calculate loss
					loss = sinkhorn_sq_batch(Xkl_imputed,p=2,niter=10,div=True,epsilon=0.01)
					mean_loss+=loss
				
				#average the loss over several batches
				mean_loss = mean_loss/n_average
			#gradients with respect to network parameters
			
			gradients = tape.gradient(loss ,Imputers[j].trainable_weights)
			opt.apply_gradients(zip(gradients,Imputers[j].trainable_weights))


			
		#after training Imputer j for k epochs update Xt using weights:
		X_no_j = tf.gather(Xt,notj,axis=1).numpy()
		X_pred = Imputers[j](X_no_j)
		Xt = Xt*maskj + X_pred*(1-maskj)
		


#save the variables for each example 
vars_to_save = [X_start, data, Xt.numpy(), mids, cids, mframe]
with open("./variables/MLP-Impute/"+name+".pickle" ,'wb') as f:
    pickle.dump(vars_to_save,f)

for j in range(len(Imputers)):
	Imputers[j].save("./variables/MLP-Impute/models/"+name+"-%i"%j)





#Impute the data using the model and plot:
# xinit=data[cids,0]
# yinit=data[cids,1]
# zinit=data[cids,2]
# #load the data that has been imputed 
# xfill=X[mids,0]
# yfill=X[mids,1]
# zfill=X[mids,2]

# #plot the 3d graph
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xinit,yinit,zinit,alpha = 0.7,color='b',marker = '.')
# ax.scatter(xfill,yfill,zfill,alpha=0.8,color ='g',marker= 'x')
# # ax.set_title("Sinkhorn Imputation %i epochs %f missing"%(epochs,per))
# ax.set_title("3d parametric imputation")

# plt.show()

