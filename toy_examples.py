"""
File: toy_examples 
Description: This file implements the data imputation via sinkorn using a batch approach 
for sklearn toy datasets.

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from utils.utils import *
from utils.sinkhorn import *
import tensorflow as tf
# import tensorflow_probability as tfp
from tqdm import tqdm
import pickle 

#this works
np.random.seed(123)

n_samples = 400 #1500

#generate the data
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples,noise=.05)
noisy_scurve,_ = datasets.make_s_curve(n_samples=n_samples)
#make 3d data 2d
noisy_scurve = np.delete(noisy_scurve,1,1)

data = noisy_moons

mframe = MissingData(data)

# create the observable dataset and mask:
mframe.MCAR_Mask(0.05) #0.15 for the other 2 


#shuffle observable data and mask
obs_data, mask = mframe.Shuffle()


print("percentage of datapoints with missing values ",mframe.percentage_missing(),"%")
print("percentage of empty points ",mframe.percentage_empty(),"%")
mids,cids = mframe.Generate_Labels()


#batchsize
batch_size = 400
assert batch_size % 2 == 0 ; "batchsize corresponds to the size of Xkl and hence must be even sized"

k = int(n_samples/batch_size)

# opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.01)
opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1)

gpu_list = get_available_gpus()

# (number of training steps , number of iterations of gradient descent)
param_list = [ (1,10000),(10000,1),(5000,100),(100,5000)]
# param_list = [ (500,1),(1,5000)]

#initialise the unobservable values
X_start = mframe.Initialise_Nans()

vars_to_save = [X_start] 

#minisise sinkhorn distance for each batch size /training epoch 
for T,epochs in param_list:
	#replace nans by means + noise 
	X = X_start.copy()

	print(" Running experiment with (T,max_epochs) = ",T,epochs)
	name = str(T)+"_"+str(epochs)+".png"

	for t in tqdm(range(T),desc = "Iteration"):
		#each part of the imputation
		indicies = [i for i in range(n_samples)]
		kl_indicies = np.random.choice(indicies,batch_size,replace=False)

		Xkl = tf.Variable(X[kl_indicies])
		msk = mask[kl_indicies]
		loss_fun = lambda: sinkhorn_sq_batch(Xkl,p=2,niter=10,div=True,epsilon=0.1)

		for e in tqdm(range(epochs),desc="RMSProp epoch"):
			#compute gradient
			grads_and_vars = opt.compute_gradients(loss_fun)

			#mask the gradient (i.e. so we are calculating w.r.t to X_impute)
			mskgrads_and_vars=[(g*(1-msk),v) for g,v in grads_and_vars]
			#apply gradient step
			opt.apply_gradients(mskgrads_and_vars)

		#update X:
		X[kl_indicies] = Xkl.numpy()



	imputed_data = np.nan_to_num(obs_data*mask) + X*(1-mask)
	vars_to_save.append(imputed_data)


	fig, axes = plt.subplots(1,2)
	axes[0].scatter(X_start[cids,0],X_start[cids,1],alpha=0.7,marker='.',label ="observable")
	axes[0].scatter(X_start[mids,0],X_start[mids,1],alpha=0.9,marker='x',color="g",label ="imputed")
	axes[0].legend()
	axes[0].set_title("initialisation")
	axes[1].scatter(imputed_data[cids,0],imputed_data[cids,1],alpha=0.7,label="observable",marker='.')
	axes[1].scatter(imputed_data[mids,0],imputed_data[mids,1],alpha=0.9,label="imputed",color="g",marker='x')
	axes[1].legend()
	axes[1].set_title("sinkhorn imputation")

	plt.savefig("./images/"+name)

	
	with open("./variables/toy_examples.pickle" ,'wb') as f:
		pickle.dump(vars_to_save,f)

#  to fix -- create a numpy.arraysplit for tensorflow for the sinkhorn algorithm