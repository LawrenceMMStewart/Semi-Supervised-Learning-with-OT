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



class MissingData():
    """
    Class that allows creation of masks and missing data
    via a variety of functions documented below.
    """
    def __init__(self,data,arr_type='float32'):
        """
        Initialises the data class and its params.

        Parameters
        ----------
        data : array n x p
        """
        #origional data and dimensions 
        self.data = data
        self.n = data.shape[0]
        self.p = data.shape[1]
        #observable data - with Nans
        self.obs_data = None
        #binary data mask 
        self.mask = None
        #type of arrays (for tensorflow compatabilities)
        self.arr_type = arr_type



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
        mask = np.random.binomial(1,1-prob,n*p).reshape((self.n,self.p)).astype(self.arr_type)

        #this could potentially generate empty data-points with probability
        # (1-p)^d. We remove this possibility here:
        for i in range(self.n):
            if (mask[i] == 0).all():
                reset_ind = np.random.randint(0,self.p)
                mask[i][reset_ind] = 1

            
        nan_mask=mask.copy()
        nan_mask[nan_mask==0]=np.nan
        obs_data=data*nan_mask

        self.obs_data = obs_data
        self.mask = mask         

        return obs_data, mask


    def Initialise_Nans(self,eta=0.1):
        """
        Sets Nans in data matrix to the sample mean for that dimension
        with some added random noise from a normal N(0,eta)

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
            for j in range(self.p):
                if np.isnan(filled[i,j]):
                    filled[i,j]=means[j]+np.random.normal(0,eta)

        return filled


    def Generate_Labels(self):
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







# #please fix bug here
# def MCAR_Mask(data,p,del_empty=False):
#     """
#     Randomly masks values of data with probability p
#     i.e. Missing Completely At Random

#     Parameters
#     -----------
#     data : array n x d
#     p : float [0,1]

#     Output
#     ----------
#     obs_data : array (n x d)
#     mask : array (n x d) with elements {0,1}
#     """

#     #extract dimensions of data
#     n=data.shape[0]
#     d=data.shape[1]

#     #randomly select some data points to change to nan (the mask)
#     mask=np.random.binomial(1,1-p,n*d).reshape(data.shape).astype('float32')
#     #create the observable data matrix
#     nan_mask=mask.copy()
#     nan_mask[nan_mask==0]=np.nan
#     obs_data=data*nan_mask

#     if del_empty:
#         #deal with the case where a datapoint is empty:
#         to_delete=[]
#         for i in range(n):
#             if np.isnan(obs_data[i]).all():
#                 to_delete.append(i)
#         #remove empty datapoints
#         obs_data=np.delete(obs_data,to_delete,axis=0)
#         mask=np.delete(mask,to_delete,axis=0)

#     return obs_data,mask


# def Initialise_Nans(data,eta=0.1):
#     """
#     Sets Nans in data matrix to the sample mean for that dimension
#     with some added random noise from a normal N(0,eta)

#     Parameters
#     ----------

#     data : array (n x d) 
#     eta : float

#     Output
#     ---------

#     filled : array (n x d)
#     """

#     filled=data.copy()
#     means=np.nanmean(data,axis=0)
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             if np.isnan(filled[i,j]):
#                 filled[i,j]=means[j]+np.random.normal(0,eta)


#     return filled




# def percentage_missing(mask):
#     """
#     Returns the percentage of the data that has missing values

#     Parameters
#     ----------
#     mask : np.array (n x d) {0,1}
    
#     Output
#     ----------
#     percentage: float

#     """

#     labels=[1 if 0 in mask[i] else 0 for i in range(len(mask))]
#     return np.mean(labels)

# def percentage_empty(mask):
#     """
#     Returns the percentage of the data that is empty
#     i.e. each point of the data is missing

#     Parameters
#     ----------
#     mask : np.array (n x d) {0,1}
    
#     Output
#     ----------
#     percentage: float

#     """
#     labels=[1 if (mask[i]==0).all() else 0 for i in range(len(mask))]
#     return np.mean(labels)

# def generate_labels(mask):
#     """
#     Returns two lists, the first consisting of the indicies
#     of the data points who are missing values, and the second
#     the indices of data points without missing values

#     Parameters
#     ----------
#     mask : np.array (n x d) {0,1}

#     Output
#     ----------
#     missing_ids : int list
#     complete_ids : int list

#     """
#     missing_ids=[]
#     complete_ids=[]
#     for i in range(len(mask)):
#         if 0 in mask[i]:
#             missing_ids.append(i)
#         else:
#             complete_ids.append(i)

#     return missing_ids,complete_ids



