"""
File: toy_examples 
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

np.random.seed(123)

parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for toy datasets")
parser.add_argument("epsilon", help = "Sinkhorn Divergance Regularisation Parameter",type = float)
parser.add_argument("exponent" , help = "Exponent of euclidean distance (1 or 2)", type = int)
parser.add_argument("dataset", help = "moons,circles or scurve", type = str )
parser.parse_args()
args = parser.parse_args()



n_samples =  400#400
#import sklearn datasets
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples,noise=.05)
noisy_scurve,_ = datasets.make_s_curve(n_samples=n_samples)
#make 3d data 2d
noisy_scurve = np.delete(noisy_scurve,1,1)

#select dataset to run on
if args.dataset == "circles":
	data = noisy_circles
elif args.dataset == "scurve":
	data = noisy_scurve
else:
	data = noisy_moons

#name for saved output
name = args.dataset+"-"+str(args.epsilon)+"-"+str(args.exponent)

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
mids,cids = mframe.Generate_ids()


#batchsize
batch_size = 50
assert batch_size % 2 == 0 ; "batchsize corresponds to the size of Xkl and hence must be even sized"

opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.1)
# opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1)
gpu_list = get_available_gpus()

#initialise the unobservable values
X_start = mframe.Initialise_Nans()
vars_to_save = [X_start] 

#minisise sinkhorn distance for each stochastic batch  
X = X_start.copy()
epochs = 10000
indicies = [i for i in range(n_samples)]

#stochastic batch gradient descent w.r.t sinkhorn divergance
for t in tqdm(range(epochs),desc = "Epoch"):
	#sample the two batches whos concatenation is Xkl
	kl_indicies = np.random.choice(indicies,batch_size,replace=True)
	Xkl = tf.Variable(X[kl_indicies])
	msk = mask[kl_indicies] #mask of Xkl
	loss_fun = lambda: sinkhorn_sq_batch(Xkl,p=args.exponent,niter=10,div=True,
		epsilon=args.epsilon) 

	#compute gradient
	grads_and_vars = opt.compute_gradients(loss_fun)

	#mask the gradient (i.e. so we are calculating w.r.t to X_impute)
	mskgrads_and_vars=[(g*(1-msk),v) for g,v in grads_and_vars]
	#apply gradient step
	opt.apply_gradients(mskgrads_and_vars)
	#update X:
	X[kl_indicies] = Xkl.numpy()

#calculate and save the imputed data
imputed_data = np.nan_to_num(obs_data*mask) + X*(1-mask)
vars_to_save.append(imputed_data)


fig, axes = plt.subplots(1,2,figsize = (12,6))
#first plot: Initialisation
axes[0].scatter(X_start[cids,0],X_start[cids,1],alpha=0.7,marker='.',label ="observable")
axes[0].scatter(X_start[mids,0],X_start[mids,1],alpha=0.9,marker='x',color="g",label ="imputed")
axes[0].legend()
axes[0].set_title("Initialisation - %f missing data"%per)
axes[0].set_facecolor('#D9E6E8')
axes[0].grid('on')
#second plot: Imputation
axes[1].scatter(imputed_data[cids,0],imputed_data[cids,1],alpha=0.7,label="observable",marker='.')
axes[1].scatter(imputed_data[mids,0],imputed_data[mids,1],alpha=0.9,label="imputed",color="g",marker='x')
axes[1].legend()
axes[1].set_title("Sinkhorn Imputation %i epochs"%epochs)
axes[1].set_facecolor('#D9E6E8')
axes[1].grid("on")


plt.savefig("./images/toy_examples/"+name+".png")


with open("./variables/toy_examples/"+name+".pickle" ,'wb') as f:
	pickle.dump(vars_to_save,f)

