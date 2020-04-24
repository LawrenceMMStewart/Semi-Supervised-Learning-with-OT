"""
File: utils
Description: Contains functions for datamasking and processing. 
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License
"""
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow_probability as tfp


@tf.function 
def sample_without_replacement(ids,batch_size):
    """
    Samples uniformly from a tensor of ids wihout replacement

    Parameters
    ----------
    ids : tensor int
    batch_size : int

    Output
    --------
    sample_ids : tensor int 
    """

    # idx = tf.random.shuffle(tf.range(len(ids)))[:batch_size]
    # sample_ids = tf.gather(ids,idx,axis=0)
    # return sample_ids
    idx = tf.random.uniform((batch_size,),0,len(ids),dtype=tf.dtypes.int32)
    sample_ids = tf.gather(ids,idx,axis=0)
    return sample_ids



@tf.function
def check_gradients(gradlist,message="Error"):
    """
    If a Nan or Inf is detected in gradlist, the list
    of gradients to check, the function will return 
    the error message

    Parameters
    -----------
    gradlist : list of tensors
    message : str (error message)

    Outputs
    ---------
    gradlist : list of tensors
    """
    gradlist = [tf.debugging.check_numerics(x,message) for x in gradlist]
    return gradlist



@tf.function
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


@tf.function
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



class MissingData():
    """
    Class that allows creation of masks and missing data
    via a variety of functions documented below.
    """
    def __init__(self,data,labels=None,arr_type='float32'):
        """
        Initialises the data class and its params.

        Parameters
        ----------
        data : array n x p
        """

        self.n = data.shape[0]
        self.m = data.shape[1]
        #observable data - with Nans
        self.obs_data = None
        #binary data mask 
        self.mask = None
        #type of arrays (for tensorflow compatabilities)
        self.arr_type = arr_type
        #data
        self.data = data.astype(arr_type)
        self.labels = labels


    def MCAR_Mask(self,prob):
        """
        Randomly masks values of data with probability p
        i.e. Missing Completely At Random

        Parameters
        -----------
        prob : float [0,1]

        Output
        ----------
        obs_data : array (n x d)
        mask : array (n x d) with elements {0,1}
        """
        mask = np.random.binomial(1,1-prob,self.n*self.m).reshape((self.n,self.m)).astype(self.arr_type)

        #this could potentially generate empty data-points with probability
        # (1-p)^d. We remove this possibility here:
        for i in range(self.n):
            if (mask[i] == 0).all():
                reset_ind = np.random.randint(0,self.m)
                mask[i][reset_ind] = 1

            
        nan_mask=mask.copy()
        nan_mask[nan_mask==0]=np.nan
        obs_data=self.data*nan_mask

        self.obs_data = obs_data
        self.mask = mask         

        return obs_data, mask



    def Initialise(self,sample):
        """
        Initialises Nans in data matrix to values of a sample
        (which can be taken from some distribution)

        Parameters
        -----------
        sample : array (n x d) values to fill the data in with
        """

        assert (self.obs_data).all!=None , "Initialisation of observable data is required"
        filled=self.obs_data.copy()

        missing = np.where(self.mask==0)
        filled[missing]=sample[missing]
        return filled


    def Initialise_Nans(self,eta=0.1):
        """
        Sets Nans in data matrix to the sample mean for that dimension
        with some added random noise from a normal N(0,eta)
        This is just a implementation of Initialize with the distribution
        being that of marcos paper.

        Parameters
        ----------
        eta : float

        Output
        ---------
        filled : array (n x d)
        """
        assert (self.obs_data).all!=None , "Initialisation of observable data is required"

        filled=self.obs_data.copy()
        means=np.nanmean(self.obs_data,axis=0)
        for i in range(self.n):
            for j in range(self.m):
                if np.isnan(filled[i,j]):
                    filled[i,j]=means[j]+np.random.normal(0,eta)

        return filled

    def Shuffle(self):
        """
        Shuffles the order of the observable data and mask 

        Output
        --------
        obs_data : array (n x p)
        mask : array (n x p) with elements in {0,1}
        labels  : (if existing) array (n x p)
        """

        if self.labels is None:        
            self.obs_data, self.mask = shuffle(self.obs_data,self.mask)
            return self.obs_data, self.mask
        else:
            self.obs_data, self.mask, self.labels = shuffle(self.obs_data,self.mask,self.labels)
            return self.obs_data, self.mask, self.labels


    def Generate_ids(self):
        """
        Returns two lists, the first consisting of the indicies
        of the data points who are missing values, and the second
        the indices of data points without missing values

        Parameters
        ----------

        Output
        ----------
        missing_ids : int list
        complete_ids : int list

        """
        assert (self.mask).all!= None , "Initialisation of the mask is required" 

        missing_ids=[]
        complete_ids=[]
        for i in range(len(self.mask)):
            if 0 in self.mask[i]:
                missing_ids.append(i)
            else:
                complete_ids.append(i)

        return missing_ids,complete_ids






    def percentage_missing(self):
        """
        Returns the percentage of the data that has missing values
        
        Output
        ----------
        percentage: float

        """
        assert (self.mask).all!= None , "Initialisation of the mask is required"

        labels=[1 if 0 in self.mask[i] else 0 for i in range(len(self.mask))]
        return np.mean(labels)


    def percentage_empty(self):
        """
        Returns the percentage of the data that is empty
        i.e. each point of the data is missing
        
        Output
        ----------
        percentage: float

        """
        assert (self.mask).all!= None , "Initialisation of the mask is required"

        labels=[1 if (self.mask[i]==0).all() else 0 for i in range(len(self.mask))]
        return np.mean(labels)




    def getbatch_ids(self, batch_size, replace = True):
        """
        Returns ids of a batch Xkl with or without replacement
        
        Parameters
        -----------
        batch_size : int (divisible by 2)
        replace : Boolean

        """
        assert (batch_size >1) and (batch_size%2==0) ; "Please enter a positive even batch_size" 
        #indicies of data
        indicies = [i for i in range(self.n)]
        #sample indicies with or without replacement
        kl_indicies = np.random.choice(indicies, batch_size, replace = replace)
        return kl_indicies


    def getbatch_jids(self, batch_size , j, replace = True):
        """
        Returns ids of a batch Xkl with or without replacement where 
        Xk has no missing values on component j and Xl has missing values
        on component j
        
        Parameters
        -----------
        batch_size : int (divisible by 2)
        j : int 
        replace : Boolean

        """
        assert (batch_size >1) and (batch_size%2==0) ; "Please enter a positive even batch_size" 
        assert j< self.m ; "Please enter a j value between 0 and d-1"
        #indicies of data with missing and full j 

        present_j = np.where(self.mask[:,j] == 1)[0]
        missing_j = np.where(self.mask[:,j] == 0)[0]
        #sample indicies with or without replacement
        k_indicies = np.random.choice(present_j, batch_size//2, replace = replace)
        l_indicies = np.random.choice(missing_j, batch_size//2, replace = replace)

        #kl indicies
        kl_indicies = np.concatenate((k_indicies,l_indicies))

        return kl_indicies



    # ------------------------------------ Simple Experiment Functions ---------------------
    def missing_secondhalf2D(self):
        """
        Creates an observable dataset and mask
        for a 2D dataset with the second half of the data missing
        """

        assert self.n%2 ==0

        obs_data=self.data.copy()
        mask=np.ones(obs_data.shape)

        #randomly replace on of the two values for each data point
        #in the second half of the dataset
        for i in range(self.n//2,self.n):
            ind=np.random.binomial(1,0.5)
            mask[i,ind]=0
            obs_data[i,ind]=np.nan

        self.obs_data = obs_data
        self.mask = mask

        return obs_data,mask


    def missing_Y_secondhalf2D(self):
        """
        Creates an observable dataset and mask
        for a 2D dataset with the second half of the data 
        having missing y co-ordinate 
        """

        assert self.n%2 ==0

        obs_data=self.data.copy()
        mask=np.ones(obs_data.shape)

        #randomly replace on of the two values for each data point
        #in the second half of the dataset
        for i in range(self.n//2,self.n):
            mask[i,1]=0
            obs_data[i,1]=np.nan

        self.obs_data = obs_data
        self.mask = mask

        return obs_data,mask



@tf.function
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
