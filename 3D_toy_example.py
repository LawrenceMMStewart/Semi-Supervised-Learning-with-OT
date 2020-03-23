"""
File: toy_examples 
Description: This file implements the data imputation via sinkorn using a batch approach 
for a 3D dataset.

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
np.random.seed(123)


parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("epsilon", help = "Sinkhorn Divergance Regularisation Parameter",type = float)
parser.add_argument("exponent" , help = "Exponent of euclidean distance (1 or 2)", type = int)
parser.parse_args()
args = parser.parse_args()

n_samples =  1500 #400
data,_ = datasets.make_s_curve(n_samples=n_samples)


#insert nans as missing data
mframe = MissingData(data)

# create the observable dataset and mask:
mframe.MCAR_Mask(0.03) #0.15 for the other 2 

#shuffle observable data and mask
obs_data, mask = mframe.Shuffle()

print("percentage of datapoints with missing values ",mframe.percentage_missing(),"%")
print("percentage of empty points ",mframe.percentage_empty(),"%")
per = mframe.percentage_missing()
#mids = positions of points that have missing values, cids = positions of points that are complete
mids,cids = mframe.Generate_Labels()


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
epochs = 50000
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

#plot 
xinit=data[cids,0]
yinit=data[cids,1]
zinit=data[cids,2]

xfill=imputed_data[mids,0]
yfill=imputed_data[mids,1]
zfill=imputed_data[mids,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xinit,yinit,zinit,alpha = 0.7,color='b',marker = '.')
ax.scatter(xfill,yfill,zfill,alpha=0.8,color ='g',marker= 'x')
plt.show()


