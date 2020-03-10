"""
File: test_SHGrad
Description: Test case to ensure gradient of sinkhorn is calculated correction

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 

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
np.random.seed(123)

#initialse two point clouds on the unit grid 

#uniform taking value in 0 -> 0.4
X = np.random.uniform(0,0.4,[50,2]).astype(np.float32)
Y = np.random.uniform(0.6,1,[50,2]).astype(np.float32)


#minimise the sinkhorn distance with respect to X 
x=tf.Variable(X)

optimizer=tf.keras.optimizers.RMSprop()

# loss_fun = lambda : sinkhorn_divergance(x,Y,euclidean_sqdist,epsilon=0.1,niter=10)

n=X.shape[0]
m=Y.shape[0]
p=2
niter=10
div=True
epsilon=0.01


loss_fun =  lambda : sinkhorn(n,m,x,Y,p,div,niter=niter,epsilon=epsilon)

losses=tfp.math.minimize(loss_fun,optimizer=optimizer,num_steps=5000,trainable_variables=[x])

print("S(X,Y) = ",sinkhorn(n,m,X,Y,p,div,niter=niter,epsilon=epsilon))
print("S(X_new,Y) = ",sinkhorn(n,m,x,Y,p,div,niter=niter,epsilon=epsilon))
print("S(Y,Y) = ",sinkhorn(n,m,Y,Y,p,div,niter=niter,epsilon=epsilon))



plt.scatter(X[:,0],X[:,1],label="X",alpha=0.8,marker='x')
plt.scatter(Y[:,0],Y[:,1],label="Y",alpha=0.8)
plt.scatter(x.numpy()[:,0],x.numpy()[:,1],label="X post gradient descent",alpha=0.8,marker='x')
plt.legend()
plt.show()





