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
    	arr_type="float32",
    	reg = 1e-4,
    	optimiser =tf.keras.optimizers.Adam(),
    	eps=0.01,
        niter = 10,
    	classifier_loss = tf.keras.losses.BinaryCrossentropy(),
        wass_reg = 1,
        p=1, #ground cost power
        ):
        """
        Initialises the data class and its params.

        Parameters
        ----------
        data : array n x p
        """
        tf.keras.backend.clear_session()
        self.n = init_data.shape[0]
        self.m = init_data.shape[1]
        #binary data mask 
        #type of arrays (for tensorflow compatabilities)
        self.arr_type = arr_type
        self.init_data = init_data.astype(self.arr_type)
        self.mask = mask.astype(self.arr_type)
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
 		wloss = sinkhorn(b1.shape[0],b2.shape[0],b1,b2,self.p,div=True,
            self.niter,self.epsilon)

        #overall loss = er_loss + wreg*wloss
        loss = er_loss + self.wass_reg*wloss

        return loss 



    def getbatch_label(self,batch_size,label,X,Y,replace=False):
        """
        returns a sample from data X which has label Y equal
        to that of the label parameter
        """
        yids = np.where(Y==label)[0]
        sample_ids = np.random.choice(yids,batch_size,replace = replace)
        batch = (X[sample_ids],Y[sample_ids])
        return batch 









