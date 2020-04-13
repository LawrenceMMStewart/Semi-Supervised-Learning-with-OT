"""
File: gradientflow.py
Description: Gradient flow between a measure consisting of 3 gaussians
one of which having larger variance than the other two 

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
import argparse
import numpy as np 
import matplotlib.pyplot as plt 


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

n_samples =  1500 #1000

µ1 = [ 0. ,0. ,0.] #mean of gaussian 1 
µ2 = [ -5. ,-5. ,-5.] # mean of gaussian 2 
µ3 = [5.,5,5]
cov = 0.1*np.eye(3)     # covariance of gaussians
cov2 = np.eye(3)

g1 = np.random.multivariate_normal(µ1,cov,n_samples//3)
g2 = np.random.multivariate_normal(µ2 ,cov,n_samples //3)
g3 = np.random.multivariate_normal(µ3 ,cov2,n_samples //3)



name = "mixturevar-"+str(args.batch_size)+'-'+str(args.epsilon)+'-'+str(args.exponent)


#batchsize
batch_size = args.batch_size
assert batch_size % 2 == 0 ; "batchsize corresponds to the size of Xkl and hence must be even sized"

opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.1)
# opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1)
gpu_list = get_available_gpus()



X = np.concatenate((g1,g2,g3),axis=0).astype(np.float32)

history= []
tracklist = [X.copy()]

# k_ops = [i for i in range(n_samples//2)]
# l_ops = [i for i in range(n_samples//2,n_samples)]
ops = [i for i in range(n_samples)]

epochs = 20000
#stochastic batch gradient descent w.r.t sinkhorn divergance
for t in tqdm(range(epochs),desc = "Epoch"):
    #sample the two batches whos concatenation is Xkl

    # k_inds = np.random.choice(k_ops, batch_size, replace = False)
    # l_inds = np.random.choice(l_ops, batch_size, replace = False)
    # kl_indicies = np.concatenate((k_inds,l_inds),axis=0)
    kl_indicies = np.random.choice(ops,batch_size,replace = False)


    Xkl = tf.Variable(X[kl_indicies])

    loss_fun = lambda: sinkhorn_sq_batch(Xkl,p=args.exponent,niter=10,div=True,
        epsilon=args.epsilon) 

    #compute gradient
    grads_and_vars = opt.compute_gradients(loss_fun)

 
    #apply gradient step
    opt.apply_gradients(grads_and_vars)
    #update X:
    X[kl_indicies] = Xkl.numpy()
    tracklist.append(X.copy())

#upon finishing training:
history.append(tracklist)

#save the variables for each example 
vars_to_save = [history]
with open("./variables/mixturevar/"+name+".pickle" ,'wb') as f:
    pickle.dump(vars_to_save,f)


