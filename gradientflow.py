"""
File: gradientflow.py
Description: This shows how points move throughout the imputation process
             using different intialisations.

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
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(12)


parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("batch_size",help = "|Xkl|",type = int)
parser.add_argument("epsilon", help = "Sinkhorn Divergance Regularisation Parameter",type = float)
parser.add_argument("exponent" , help = "Exponent of euclidean distance (1 or 2)", type = float)
parser.parse_args()
args = parser.parse_args()

n_samples =  1000 #1000
data,_ = datasets.make_s_curve(n_samples=n_samples)


name = str(args.batch_size)+'-'+str(args.epsilon)+'-'+str(args.exponent)

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


#batchsize
batch_size = args.batch_size
assert batch_size % 2 == 0 ; "batchsize corresponds to the size of Xkl and hence must be even sized"

opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.1)
# opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1)
gpu_list = get_available_gpus()


#create two different initialisations for the imputations:

#1) Initialisation with normal distribution around means
X_start1 = mframe.Initialise_Nans()

#2) Initialisation with mixture of normals 
µ1 = [0.3,0.3,0.3] #means of gaussians
µ2 = [0.7,0.7,0.7]
means = [µ1,µ2]

cov = [[0.1,0,0],[0,0.1,0],[0,0,0.1]] #covarience


sample = np.zeros(data.shape)
for i in range(n_samples):
    gid = np.random.randint(0,2) # choose which gaussian to sample from
    sample[i] = np.random.multivariate_normal(means[gid],cov)

#initialise with this mixture
X_start2 = mframe.Initialise(sample)

inits = [X_start1,X_start2]
epochs = 20000
indicies = [i for i in range(n_samples)]

history= []

#for both initialisations
for X_start in inits:


    #minisise sinkhorn distance for each stochastic batch  
    X = X_start.copy()
    tracklist= [X.copy()]

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
        tracklist.append(X.copy())

    #upon finishing training:
    history.append(tracklist)

#save the variables for each example 
vars_to_save = [history,data,mids,cids]
with open("./variables/gradientflow/"+name+".pickle" ,'wb') as f:
    pickle.dump(vars_to_save,f)
breakpoint()

