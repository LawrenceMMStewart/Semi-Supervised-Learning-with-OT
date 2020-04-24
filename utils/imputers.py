"""
File: imputers
Description: Contains functions and classes for training k imputers
for k dimensional observable data in parallel with training a classifier
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License
"""



import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from utils.utils import *
from utils.sinkhorn import *
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
import pickle 


class ClassifyImpute():
    """
    Class containing imputers and a classifer, allowing
    training of classifier and imputers in parallel

    """
    def __init__(self,init_data,labels,mask,imputers=None,
        classifier = None,
        # arr_type="float32",
        reg = 1e-4,
        optimiser =tf.keras.optimizers.Adam(),
        eps=0.01,
        niter = 10,
        classifier_loss = tf.keras.losses.BinaryCrossentropy(),
        wass_reg = 1,
        p=1, 
        ):
        """
        Initialises the data class and its params.

        Parameters
        ----------
        init_data : tensor n x m
        labels : tensor n x 1
        mask  : binary tensor n x m
        imputers : tf.model list
        reg : float
        optimiser : tf.optimizer
        eps : float (sinkhorn parameter)
        niter : int (sinkhorn number of iterations)
        classifier_loss : tf.loss function
        wass_reg : float for regularisation term
        p : float (ground cost power)


        Output
        --------
        Creates class of imputers and classifier
        """
       
        self.n = init_data.shape[0]
        self.m = init_data.shape[1]

        self.init_data = init_data
        self.mask = mask
        self.labels = labels
        self.imputers = []
        self.classifier = []


        self.opt = optimiser
        self.reg =reg
        self.eps = eps
        self.classifier_loss = classifier_loss
        self.wass_reg = wass_reg
        self.p = p
        self.niter = niter


        self.mids,self.cids = self.Generate_ids()

        self.losshist = []
        self.gradhist = [] 
        #imputers
        if imputers is None:
            for i in range(self.m):

                #add m MLP with relu units and L2 regularisation
                self.imputers.append(tf.keras.Sequential([
                    keras.layers.Dense(2*(self.m-1), activation ='relu',input_shape = (self.m-1,),
                        kernel_regularizer=tf.keras.regularizers.l2(reg)),
                    keras.layers.Dense(self.m-1,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(reg)),
                    keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(reg))]))

        #if custom imputers are defined
        else:
            self.imputers = imputers


        #classifier
        if classifier is None:
            self.classifier = tf.keras.Sequential([keras.layers.Dense(20,activation = 'relu', input_shape = (self.m,), 
                kernel_regularizer =tf.keras.regularizers.l2(self.reg)),
            keras.layers.Dense(5,activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(self.reg)),
            keras.layers.Dense(1, activation = 'sigmoid')])
        else:
            self.classifier = classifier






    #   _____                       _ _             
    #  / ____|                     | (_)            
    # | (___   __ _ _ __ ___  _ __ | |_ _ __   __ _ 
    #  \___ \ / _` | '_ ` _ \| '_ \| | | '_ \ / _` |
    #  ____) | (_| | | | | | | |_) | | | | | | (_| |
    # |_____/ \__,_|_| |_| |_| .__/|_|_|_| |_|\__, |
    #                        | |               __/ |
    #                        |_|              |___/ 




    @tf.function 
    def getbatch_label(self,batch_size,X,Y,label):
        """
        samples a batch of size_batch size from data (X,Y)
        conditioning on the value of label

        Parameters
        ----------
        batch_size : int
        X : tensor n x m
        Y : tensor n x 1
        label : int

        Output
        ---------
        tuple : (tensor : batch_size x m , tensor: batch_size, 1)
        """
        yids = tf.where(Y==label)[:,0]
        sample_ids = sample_without_replacement(yids,batch_size)
        batchX = tf.gather(X,sample_ids,axis=0)
        batchY = tf.gather(Y,sample_ids,axis=0)
        return (batchX,batchY)

    @tf.function
    def getbatch_label_MC(self,batch_size,X,Y,label,j,missing=True):
        """
        samples a batch of size batch_size from data (X,Y)
        conditioning on the value of label and conditioning on
        whether the data should be missing or complete (controlled
        by the missing boolean parameter) on dimension j.

        Parameters
        ----------
        batch_size : int
        X : tensor n x m
        Y : tensor n x 1
        label : int
        j : int in range 0,m-1
        missing : boolean

        Output
        ---------
        tuple : (tensor : batch_size x m , tensor: batch_size, 1)
        """
        assert j< self.m ; "Please enter a j value between 0 and d-1"
        
        if missing:
            #indicies where mask is missing on dimension j 
            jids = tf.where(self.mask[:,j] == 0)[:,0]
        if not missing:
            jids = tf.where(self.mask[:,j] == 1)[:,0]
        #indicies where y=label
        yids = tf.where(Y==label)[:,0]
        
  
        # to calculate the intersection of jids and yids we use a set 
        tset = tf.sets.intersection(jids[None,:], yids[None, :])
        ids = tset.values
        #sample points from the possible ids
        sample_ids = sample_without_replacement(ids,batch_size)
        batchX = tf.gather(X,sample_ids,axis=0)
        batchY = tf.gather(Y,sample_ids,axis=0)
        return (batchX,batchY)
      

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

        missing_ids=[]
        complete_ids=[]
        for i in range(len(self.mask)):
            if 0 in self.mask[i]:
                missing_ids.append(i)
            else:
                complete_ids.append(i)

        return missing_ids,complete_ids


    def Generate_label_ids(self):
        """
        Generates ids for missing and complete points
        conditioned on labels (used for plotting)

        Outputs
        --------
        mids0 : list of ids of missing points with label equal to 0
        mids1 : list of ids of missing points with label equal to 1
        cids0 : list of ids of complete points with label equal to 0
        cids1 : list of ids of complete points with label equal to 1
        """
        
        label0 = np.where(self.labels==0)[0]
        label1 = np.where(self.labels==1)[0]

        #create mids0 and mids1 
        mids0 = np.intersect1d(self.mids,label0)
        mids1 = np.intersect1d(self.mids,label1)

        #create cids0 and cids1
        cids0 = np.intersect1d(self.cids,label0)
        cids1 = np.intersect1d(self.cids,label1)

        return mids0,mids1,cids0,cids1




    #  _______        _       _             
    # |__   __|      (_)     (_)            
    #    | |_ __ __ _ _ _ __  _ _ __   __ _ 
    #    | | '__/ _` | | '_ \| | '_ \ / _` |
    #    | | | | (_| | | | | | | | | | (_| |
    #    |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
    #                                  __/ |
    #                                 |___/ 




    @tf.function
    def impute(self,j,X):
        """
        Impute dimension j of X  
        """
        #create a mask for dimension j (all other values are considered observable)
        #and call this mask maskj
        masklist = []
        for i in range(self.m):
            if i==j:
                masklist.append(tf.gather(self.mask,1,axis=1))
            else:
                masklist.append(tf.ones(self.n))

        
        maskj =tf.stack(masklist,axis=1)


        #indicies of dimensions that are not j
        notj = tf.constant([i for i in range(self.m) if i!=j])


        X_no_j = tf.gather(X,notj,axis=1)
        X_pred = self.imputers[j](X_no_j)
        X_hat = X*maskj + X_pred*(1-maskj)

        return X_hat



   
    # def train_imputer(self,j,X,Y,batch_size,epochs=1,disable_bar=False):
    #     """
    #     Train imputer j for niter iterations 
    #     """
    #     if epochs==1:
    #         disable_bar=True
    #     for i in tqdm(range(epochs),desc = "Imputer %i"%j,disable = disable_bar):
    #         #train on first label y=0
    #         with tf.GradientTape() as tape:
    #             Xhat = self.impute(j,X)
    #             #batch of complete values
    #             bc,_ = self.getbatch_label_complete(batch_size,0,j,Xhat,Y)
    #             bm,_ = self.getbatch_label_missing(batch_size,0,j,Xhat,Y)
    #             #run sinkhorn on the batches
    #             loss = sinkhorn(bc.shape[0],bm.shape[0],bc,bm,self.p,div=True,niter=self.niter,epsilon=self.eps)

    #             # self.losshist.append(loss.numpy())
    #         #perform gradient step on NN for dim j 

    #         # tf.debugging.Assert(loss>0,[loss])
    #         gradients = tape.gradient(loss ,self.imputers[j].trainable_weights)
    #         gradients = check_gradients(gradients)
    #         self.opt.apply_gradients(zip(gradients,self.imputers[j].trainable_weights))

    #         # self.gradhist.append(gradients)

    #         #train on second label y=1
    #         with tf.GradientTape() as tape:
    #             Xhat = self.impute(j,X)
    #             #batch of complete values
    #             bc,_ = self.getbatch_label_complete(batch_size,1,j,Xhat,Y)
    #             bm,_ = self.getbatch_label_missing(batch_size,1,j,Xhat,Y)
    #             #run sinkhorn on the batches
    #             loss = sinkhorn(bc.shape[0],bm.shape[0],bc,bm,self.p,div=True,niter=self.niter,epsilon=self.eps)
    #             # self.losshist.append(loss.numpy())

    #         # tf.debugging.Assert(loss>0,[loss])
    #         #perform gradient step on NN for dim j 
    #         gradients = tape.gradient(loss ,self.imputers[j].trainable_weights)
    #         gradients = check_gradients(gradients)
    #         self.opt.apply_gradients(zip(gradients,self.imputers[j].trainable_weights))

    #         # self.gradhist.append(gradients)
        

    @tf.function
    def forward(self,batch,wbatch1,wbatch2):
        """
        Forward of learning batch + wasserstein distance of two batches
        batch = tuple (data,labels)
        """

        #unpack labels for ERM batch and wasserstein batch
        X,Y = batch
        b1,b1_labels = wbatch1
        b2,b2_labels = wbatch2 

        #Predict and score ERM 
        out = self.classifier(X)
        er_loss = self.classifier_loss(out,Y)

        #wasserstein regularisation
        wloss = sinkhorn(b1.shape[0],b2.shape[0],b1,b2,self.p,div=True,niter=self.niter,epsilon=self.eps)

        #overall loss = er_loss + wreg*wloss
        loss = er_loss + self.wass_reg*wloss

        return loss 







