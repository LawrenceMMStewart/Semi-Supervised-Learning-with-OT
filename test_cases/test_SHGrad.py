"""
Test case to ensure gradient of sinkhorn is calculated correction
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Liscence: Mit License 


Consider the problem of 2 uniform point clouds, running sinkhorn w.r.t. one of them and
optimising via gradient descent we would expect them to overlap
"""

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from utils.sinkhorn import *    # neither of these work -- need to cc up one to run it 
from utils.utils import *
import tensorflow_probability as tfp
from tqdm import tqdm


#initialse two point clouds on the unit grid 

#uniform taking value in 0 -> 0.4
X = np.random.uniform(0,0.4,[30,2])
Y = np.random.uniform(0.6,1,[30,2])


#minimise the sinkhorn distance with respect to X 
x=tf.Variable(X)

optimizer=tf.keras.optimizers.Adam()

loss_fun = lambda : sinkhorn_divergance(x,Y,euclidean_sqdist,epsilon=1000,niter=1000)
losses=tfp.math.minimize(loss_fun,optimizer=optimizer,num_steps=10000)

print("S(X,Y) = ",sinkhorn_divergance(X,Y,euclidean_sqdist,epsilon=1000,niter=1000))
print("S(X_new,Y) = ",sinkhorn_divergance(x,Y,euclidean_sqdist,epsilon=1000,niter=1000))
print("S(Y,Y) = ",sinkhorn_divergance(Y,Y,euclidean_sqdist,epsilon=1000,niter=1000))

plt.scatter(X[:,0],X[:,1],label="X",alpha=0.8,marker='x')
plt.scatter(Y[:,0],Y[:,1],label="Y",alpha=0.8)
plt.scatter(x.numpy()[:,0],x.numpy()[:,1],label="X post gradient descent",alpha=0.8,marker='x')
plt.legend()
plt.show()





