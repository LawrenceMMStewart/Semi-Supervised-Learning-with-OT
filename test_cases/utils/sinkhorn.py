"""
Sinkhorn Algorithm for regularised optimal transport
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Liscence: Mit License
"""

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from utils import * 
np.random.seed(123)


def sinkhorn_loss(x,y,Cost,epsilon=0.01,niter=10,err=1e-1):
    """
    Performs sinkhorn algorithm on two measures using the log-exp
    trick for numerical stability, returning the sinkhorn loss
    
    Parameters
    ----------
    x : tf.tensor n x d 
    y : tf.tensor m x d 
    Cost : function to evaluate pairwise cost

    Output
    --------

    Sinkhorn Loss: tf.tensor 

    """
    #Create the cost matrix
    C=Cost(x,y)

    n=x.shape[0]
    m=y.shape[0]

    #logsum lambda function:
    lsexp = lambda x : tf.math.reduce_logsumexp(x,axis=1,keepdims=True)

    def M(u,v):
        """
        Creates the stable sinkhorn matrix M_ij=( - C_ij + u_i + v_j) / e 
        """ 
        return (-C + tf.expand_dims(u,1) + tf.expand_dims(v,0) )/ epsilon
    
    
    #define the uniform weightings
    µ1=tf.ones(n,dtype=tf.float32)/n
    µ2=tf.ones(m,dtype=tf.float32)/m

    #initialise u,v
    u,v= (0.*µ1 ,0.*µ2)
    u1=0.*µ1
    norm=2*err


    def step(u,v,u1,norm):
    #perform the fixed point iteration:
        u1=u
        u=epsilon * (tf.math.log(µ1) - tf.squeeze(lsexp(M(u,v))))+u
        v=epsilon * (tf.math.log(µ2) - tf.squeeze(lsexp(tf.transpose(M(u,v)))))+v
        norm=tf.reduce_sum(tf.abs(u-u1))
        return u,v,u1,norm

    def condition(u,v,u1,norm):
        return norm<err

    u,v,_,_=tf.while_loop(condition,step,[u,v,u1,norm],maximum_iterations=niter)

    #calculate optimal transport plan
    P=tf.math.exp(M(u,v))
    #return sinkhorn loss
    sloss=tf.reduce_sum(P*C)


    return sloss



def sinkhorn_divergance(x,y,Cost,epsilon=0.01,niter=10):
    """
    Returns the sinkhorn divergance for two poinclouds x and y 
    (can generalise both this function and the above to non-uniform measures)
    """   
    sxy=sinkhorn_loss(x,y,Cost,epsilon=epsilon,niter=niter)
    syy=sinkhorn_loss(y,y,Cost,epsilon=epsilon,niter=niter)
    sxx=sinkhorn_loss(x,x,Cost,epsilon=epsilon,niter=niter) 

    return sxy-0.5*(sxx+syy)



# @tf.function
# def split_tensor(X,K):
#     return tf.numpy_function(np.array_split,[X,K],Tout=tf.float32)




def sinkhorn_sq_batch(X,epsilon=0.01,niter=10):
    #split batch in half
    X1,X2 = tf.split(X,2)
    #caluclate sinkhorn divergance
    return sinkhorn_divergance(X1,X2,euclidean_sqdist,epsilon=epsilon,niter=niter)


if __name__=="__main__":

    #run a simple experiment to check it is working

    from utils import *

    N = [300,200]
    d = 2
    x = np.random.rand(2,N[0])-.5

    theta = 2*np.pi*np.random.rand(1,N[1])
    r = .8 + .2*np.random.rand(1,N[1])
    y = np.vstack((np.cos(theta)*r,np.sin(theta)*r))
    plotp = lambda x,col: plt.scatter(x[0,:], x[1,:], s=200, edgecolors="k", c=col, linewidths=2)

    x=tf.constant(x.T)
    y=tf.constant(y.T)
    
    with tf.GradientTape() as t:
        t.watch(x)
        s=sinkhorn_loss(x,y,euclidean_sqdist,niter=3000)
        print(s)
        print(t.gradient(s,x))