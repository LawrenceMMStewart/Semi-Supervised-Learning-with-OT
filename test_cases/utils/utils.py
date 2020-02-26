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


# @tf.function(input_signature=[tf.TensorSpec(tf.float32,tf.int32)]) 
# def tensor_split(X,K):
#     """
#     Splits a tensor X into k tensors along axis=0 as best as possible

#     Parameters
#     -----------
#     X : float32 tensor (n x d) 
#     k : tensor


#     Output
#     ----------
#     L : List of tensors

#     """
#     return tf.numpy_function(np.array_split,[X,K],Tout=tf.float32)


#please fix bug here
def MCAR_Mask(data,p,del_empty=False):
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

    if del_empty:
        #deal with the case where a datapoint is empty:
        to_delete=[]
        for i in range(n):
            if np.isnan(obs_data[i]).all():
                to_delete.append(i)
        #remove empty datapoints
        obs_data=np.delete(obs_data,to_delete,axis=0)
        mask=np.delete(mask,to_delete,axis=0)

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




def percentage_missing(mask):
    """
    Returns the percentage of the data that has missing values

    Parameters
    ----------
    mask : np.array (n x d) {0,1}
    
    Output
    ----------
    percentage: float

    """

    labels=[1 if 0 in mask[i] else 0 for i in range(len(mask))]
    return np.mean(labels)

def percentage_empty(mask):
    """
    Returns the percentage of the data that is empty
    i.e. each point of the data is missing

    Parameters
    ----------
    mask : np.array (n x d) {0,1}
    
    Output
    ----------
    percentage: float

    """
    labels=[1 if (mask[i]==0).all() else 0 for i in range(len(mask))]
    return np.mean(labels)

def generate_labels(mask):
    """
    Returns two lists, the first consisting of the indicies
    of the data points who are missing values, and the second
    the indices of data points without missing values

    Parameters
    ----------
    mask : np.array (n x d) {0,1}

    Output
    ----------
    missing_ids : int list
    complete_ids : int list

    """
    missing_ids=[]
    complete_ids=[]
    for i in range(len(mask)):
        if 0 in mask[i]:
            missing_ids.append(i)
        else:
            complete_ids.append(i)

    return missing_ids,complete_ids



