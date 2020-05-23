"""
Sinkhorn Algorithm for regularised optimal transport
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Functions taken from Claici et al. sclaici@csail.mit.edu
Liscence: Mit License
"""

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt




def cost_mat(X,Y,n,m,p):
    """
    Returns table of pointwise Eucildean Distances
    C_ij=|| x_i - y_j ||^p

    Parameters
    ----------
    X : (tensor) (n x p) 
    Y : (tensor) (m x p)
    n : int
    m : int 
    p : int

    Output
    ---------
    C : (tensor) (n x m) 
    """

    #sum over second dimension for each
    X2=tf.reduce_sum(X**2,1) # (n,) 
    Y2=tf.reduce_sum(Y**2,1) # (m,) 

    #add axis 
    X2=tf.expand_dims(X2,1)  #(n ,1)
    Y2=tf.expand_dims(Y2,0) # (1,m)

    #broadcast:
    X2=tf.tile(X2,[1,m])
    Y2=tf.tile(Y2,[n,1])

    C = X2 +Y2-2.*X@tf.transpose(Y)
    return tf.math.sqrt(C+10**(-6))**p


def K_tild(u,v,n,m,C,epsilon):
    """
    Calculates the matrix  ktild = exp (1/eps ( ui+vj-Cij) ) for sinkhorn step
    for uniform measures
    """
    C_tild = C - tf.transpose(tf.reshape(tf.tile(u[:,0],[m]),[m,n])) - tf.reshape(tf.tile(v[:,0],[n]),[n,m])
    k_tild = tf.exp(-C_tild/epsilon)
    return k_tild


def sinkhorn_step_log(u,v,n,m,C,epsilon):
    """
    Calculates one step of sinkhorn in logsumexp manner
    for uniform measures
    """
    mu = tf.cast(1/n, tf.float32)
    nu = tf.cast(1/m, tf.float32)
    Ku = tf.reshape( tf.reduce_sum(K_tild(u,v,n,m,C,epsilon),axis = 1) ,[n,1] )
    u = epsilon*(tf.math.log(mu) - tf.math.log(Ku +10**(-6))) + u 
    Kv = tf.reshape( tf.reduce_sum(K_tild(u,v,n,m,C,epsilon),axis = 0), [m,1] )
    v = epsilon*(tf.math.log(nu) - tf.math.log(Kv +10**(-6))) + v 
    return u,v


def sinkhorn_cost_log(n,m,C,niter,epsilon):
    """
    Calculates the log sinkhorn cost for uniform measures 
    """
    
    u = tf.zeros([n,1])
    v = tf.zeros([m,1])
    for i in tf.range(niter):
        if i!=niter-1:
            u,v =sinkhorn_step_log(u,v,n
                ,m,C,epsilon)
            u =tf.stop_gradient(u)
            v =tf.stop_gradient(v)
        else:
            u,v = sinkhorn_step_log(u,v,n
                ,m,C,epsilon)
        # u,v = sinkhorn_step_log(u,v,n,m,C,epsilon)

    gamma_log = K_tild(u,v,n,m,C,epsilon)
    final_cost_log = tf.reduce_sum(gamma_log*C)
    return final_cost_log
    
@tf.function
def sinkhorn(X,Y,
    p=tf.constant(1.0,dtype=tf.float32),
    div=tf.constant(True),
    niter=tf.constant(150),
    epsilon=tf.constant(0.1,dtype=tf.float32)):
    """
    Returns the sinkhorn loss or divergance calculated 
    using the logsumexp approach for uniform measures

    Parameters
    ----------
    X,Y : tensor (n x p), (m x p)
    p : tf.float32 (exponent)
    div : tf.Boolean
    niter : tf.int
    epsilon: tf.float32


    Output
    ---------
    sinkhorn_costXY : tensor (1)
    """
    n = X.shape[0]
    m = Y.shape[0]
    CXY = cost_mat(X,Y,n,m,p)
    sinkhorn_costXY = sinkhorn_cost_log(n,m,CXY,niter,epsilon)
    if div:
        CXX = cost_mat(X,X,n,n,p)
        CYY = cost_mat(Y,Y,m,m,p)
        sinkhorn_costXX = sinkhorn_cost_log(n,n,CXX,niter,epsilon)
        sinkhorn_costYY = sinkhorn_cost_log(m,m,CYY,niter,epsilon)
        return sinkhorn_costXY - 1/2 *(sinkhorn_costXX+sinkhorn_costYY)
    else:
        return sinkhorn_costXY

 

