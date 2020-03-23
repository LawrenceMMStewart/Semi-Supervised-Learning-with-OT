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
from utils import *


@tf.function 
def cost_mat(X,Y,n,m,p=2):
    """
    Returns table of pointwise Eucildean Distances
    C_ij=|| x_i - y_j ||^2

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
    XX = tf.reduce_sum(tf.multiply(X,X),axis=1)
    YY = tf.reduce_sum(tf.multiply(Y,Y),axis=1)
    C1 = tf.transpose(tf.reshape(tf.tile(XX,[m]),[m,n]))
    C2 = tf.reshape(tf.tile(YY,[n]),[n,m])
    C3 = tf.transpose(tf.linalg.matmul(Y,tf.transpose(X)))
    C = C1 + C2 - 2*C3;
    if p == 2:
        return C
    else:
        return tf.sqrt(C+10**(-3))**p

@tf.function 
def K_tild(u,v,n,m,C,epsilon=0.1):
    """
    Calculates the matrix exp ( C_ij -ui- vj ) for sinkhorn step
    for uniform measures
    """
    C_tild = C - tf.transpose(tf.reshape(tf.tile(u[:,0],[m]),[m,n])) - tf.reshape(tf.tile(v[:,0],[n]),[n,m])
    k_tild = tf.exp(-C_tild/epsilon)
    return k_tild

@tf.function 
def sinkhorn_step_log(u,v,n,m,C,epsilon=0.1):
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

@tf.function 
def sinkhorn_cost_log(n,m,C,niter,epsilon=0.1):
    """
    Calculates the log sinkhorn cost for uniform measures 
    """
    
    u = tf.zeros([n,1])
    v = tf.zeros([m,1])
    for i in range(niter):
        u,v = sinkhorn_step_log(u,v,n,m,C,epsilon)
    gamma_log = K_tild(u,v,n,m,C,epsilon)
    final_cost_log = tf.reduce_sum(gamma_log*C)
    return final_cost_log

def sinkhorn(n,m,X,Y,p,div,niter=10,epsilon=0.1):
    """
    Returns the sinkhorn loss or divergance calculated 
    using the logsumexp approach for uniform measures

    Parameters
    ----------
    n,m : int
    p : float (exponent)
    X,Y : tensor (n x p), (m x p)
    div : Boolean
    niter : int
    epsilon: float


    Output
    ---------
    sinkhorn_costXY : tensor (1)
    """

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

 

def sinkhorn_sq_batch(X,p=2,niter=10,div=True,epsilon=0.1):
    #split batch in half
    X1,X2 = tf.split(X,2)
    n=X1.shape[0]
    m=X2.shape[0]
    #caluclate sinkhorn divergance
    return sinkhorn(n,m,X1,X2,p,div,niter,epsilon)

