"""
Functions for Regularised OT 
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Liscence: Mit License
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(X,Y):
    """
    Returns table of pointwise Eucildean Distances
    C_ij=|| x_i - y_j ||

    Parameters
    ----------
    X : (tensor) n x d 
    Y : (tensor) m x d

    Output
    ---------
    C : (tensor) n x m 

    """

    #obtain number of samples for both X and Y 
    n=X.shape[0]
    m=Y.shape[0]

    #sum over second dimension for each
    X2=tf.reduce_sum(X**2,1) # (n,) 
    Y2=tf.reduce_sum(Y**2,1) # (m,) 

    #add axis 
    X2=tf.expand_dims(X2,1)  #(n ,1)
    Y2=tf.expand_dims(Y2,0) # (1,m)

    #broadcast:
    X2=tf.tile(X2,[1,m])
    Y2=tf.tile(Y2,[n,1])

    return tf.cast(tf.math.sqrt(X2+Y2-2.*X@tf.transpose(Y)),tf.float32)





def euclidean_sqdist(X,Y):
    """
    Returns table of pointwise Eucildean Distances
    C_ij=|| x_i - y_j ||^2

    Parameters
    ----------
    X : (tensor) (n x d) 
    Y : (tensor) (m x d)

    Output
    ---------
    C : (tensor) (n x m) 

    """

    #obtain number of samples for both X and Y 
    n=X.shape[0]
    m=Y.shape[0]

    #sum over second dimension for each
    X2=tf.reduce_sum(X**2,1) # (n,) 
    Y2=tf.reduce_sum(Y**2,1) # (m,) 

    #add axis 
    X2=tf.expand_dims(X2,1)  #(n ,1)
    Y2=tf.expand_dims(Y2,0) # (1,m)

    #broadcast:
    X2=tf.tile(X2,[1,m])
    Y2=tf.tile(Y2,[n,1])

    return tf.cast(X2+Y2-2.*X@tf.transpose(Y),tf.float32)




def MCAR_Mask(data,p):
    """
    Randomly masks values of data with probability p
    i.e. Missing Completely At Random

    Parameters
    -----------
    data : array n x d
    p : float [0,1]

    Output
    ----------
    obs_data : array (n x d)
    mask : array (n x d) with elements {0,1}
    """

    #extract dimensions of data
    n=data.shape[0]
    d=data.shape[1]

    #randomly select some data points to change to nan (the mask)
    mask=np.random.binomial(1,1-p,n*d).reshape(data.shape).astype('float32')
    #create the observable data matrix
    nan_mask=mask.copy()
    nan_mask[nan_mask==0]=np.nan
    obs_data=data*nan_mask

    #deal with the case where a datapoint is empty:
    for i in range(n):
        if (obs_data[i]==np.nan).all():
            obs_data=np.delete(obs_data,(i),axis=0)
            mask=np.delete(mask,(i),axis=0)

    return obs_data,mask


def Initialise_Nans(data,η=0.1):
    """
    Sets Nans in data matrix to the sample mean for that dimension
    with some added random noise from a normal distribution

    Parameters
    ----------

    data : array (n x d) 
    η : float

    Output
    ---------

    filled : array (n x d)
    """

    filled=data.copy()
    means=np.nanmean(data,axis=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(filled[i,j]):
                filled[i,j]=means[j]+np.random.normal(0,η)


    return filled




