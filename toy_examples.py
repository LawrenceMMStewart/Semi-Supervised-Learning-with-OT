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
import tensorflow_probability as tfp
from tqdm import tqdm


np.random.seed(123) 

n_samples = 500 #1500

#generate the data
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples,noise=.05)
noisy_scurve,_ = datasets.make_s_curve(n_samples=n_samples)
#make 3d data 2d
noisy_scurve = np.delete(noisy_scurve,1,1)

data = noisy_moons

mframe = MissingData(data)

# create the observable dataset and mask:
mframe.MCAR_Mask(0.10) #0.15 for the other 2 


#shuffle observable data and mask
obs_data, mask = mframe.Shuffle()


print("percentage of datapoints with missing values ",mframe.percentage_missing(),"%")
print("percentage of empty points ",mframe.percentage_empty(),"%")
mids,cids = mframe.Generate_Labels()


#batchsize
batch_size = 250
assert batch_size % 2 == 0 ; "batchsize corresponds to the size of Xkl and hence must be even sized"

k = int(n_samples/batch_size)

#replace nans by means + noise 
X = mframe.Initialise_Nans()


T=100
epochs = 500
opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.01)

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


imputed_data = np.nan_to_num(obs_data*mask)+X*(1-mask)



fig, axes = plt.subplots(1,2)
axes[0].scatter(noisy_scurve[:,0],noisy_scurve[:,1])
axes[0].set_title("ground truth")
axes[1].scatter(data[cids,0],data[cids,1],alpha=0.8,label="observable")
axes[1].scatter(imputed_data[mids,0],imputed_data[mids,1],alpha=0.5,label="imputed",marker='x')
axes[1].legend()
axes[1].set_title("imputation")
plt.show()
    

#  to fix -- create a numpy.arraysplit for tensorflow for the sinkhorn algorithm