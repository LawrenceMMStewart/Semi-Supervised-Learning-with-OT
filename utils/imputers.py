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
# from tensorflow import keras
import pickle 


class ClassifyImpute():
    """
    Class containing imputers and a classifer, allowing
    training of classifier and imputers in parallel

    """
    def __init__(self,X0,labels,mask,imputers=None,
        classifier = None,
        batch_size = 100,
        reg = 1e-4,
        optimiser =tf.keras.optimizers.Adam(),
        eps=0.01,
        niter = 100,
        classifier_loss = tf.keras.losses.BinaryCrossentropy(),
        wass_reg = 1,
        p=1, 
        ):
        """
        Initialises the data class and its params.

        Parameters
        ----------
        X0 : tensor n x m
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
       
        self.n = X0.shape[0]
        self.m = X0.shape[1]

        self.X0 = X0
        self.Xt = tf.identity(X0)
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
        self.batch_size = batch_size
        # self.losshist = []
        # self.gradhist = [] 
        


        #  _____                       _                
        # |_   _|                     | |               
        #   | |  _ __ ___  _ __  _   _| |_ ___ _ __ ___ 
        #   | | | '_ ` _ \| '_ \| | | | __/ _ \ '__/ __|
        #  _| |_| | | | | | |_) | |_| | ||  __/ |  \__ \
        # |_____|_| |_| |_| .__/ \__,_|\__\___|_|  |___/
        #                 | |                           
        #                 |_|                           



        if imputers is None:
            for i in range(self.m):

                #add m MLP with relu units and L2 regularisation
                self.imputers.append(tf.keras.Sequential([
                    tf.keras.layers.Dense(2*(self.m-1), activation ='relu',input_shape = (self.m-1,),
                        kernel_regularizer=tf.keras.regularizers.l2(reg)),
                    tf.keras.layers.Dense(self.m-1,activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(reg)),
                    tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(reg))]))

        #if custom imputers are defined
        else:
            self.imputers = imputers


        #classifier
        if classifier is None:
            self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(20,activation = 'relu', input_shape = (self.m,), 
                kernel_regularizer =tf.keras.regularizers.l2(self.reg)),
            tf.keras.layers.Dense(5,activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(self.reg)),
            tf.keras.layers.Dense(1, activation = 'sigmoid')])
        else:
            self.classifier = classifier




        #  _____                                          _        _   _                 
        # |  __ \                                        | |      | | (_)                
        # | |__) | __ ___  ___ ___  _ __ ___  _ __  _   _| |_ __ _| |_ _  ___  _ __  ___ 
        # |  ___/ '__/ _ \/ __/ _ \| '_ ` _ \| '_ \| | | | __/ _` | __| |/ _ \| '_ \/ __|
        # | |   | | |  __/ (_| (_) | | | | | | |_) | |_| | || (_| | |_| | (_) | | | \__ \
        # |_|   |_|  \___|\___\___/|_| |_| |_| .__/ \__,_|\__\__,_|\__|_|\___/|_| |_|___/
        #                                    | |                                         
        #                                    |_|                                         



        #maskj considers all missing points not on axis j to be observed
        self.maskj_list = []
        self.make_maskj_list()

        #for use in imputation function (all the axis which are not j)
        self.notj_list = []
        for j in range(self.m):
            #indicies of dimensions that are not j
            notj = tf.constant([i for i in range(self.m) if i!=j])
            self.notj_list.append(notj)

        #All possible combinations of Xt without its j'th axis
        self.Xt_no_j_list = []
        for j in range(self.m):
            Xt_no_j = tf.gather(self.Xt,self.notj_list[j],axis=1)
            self.Xt_no_j_list.append(Xt_no_j)

        #label ids of where labels==0 or labels ==1
        self.labels_ids_list = []
        for y in range(2):
            self.labels_ids_list.append(tf.where(self.labels==y)[:,0])

        #ids where m is missing on axis j
        self.mj_ids_list = []
        for j in range(self.m):
            self.mj_ids_list.append(tf.where(self.mask[:,j]==0)[:,0])


        self.mjy_ids_list = []
        for y in range(2):
            tempy = []
            for j in range(self.m):
                tset = tf.sets.intersection(self.labels_ids_list[y][None,:],
                    self.mj_ids_list[j][None, :])
                tempy.append(tset.values)
            self.mjy_ids_list.append(tempy)



        #  _____        _                 _       
        # |  __ \      | |               | |      
        # | |  | | __ _| |_ __ _ ___  ___| |_ ___ 
        # | |  | |/ _` | __/ _` / __|/ _ \ __/ __|
        # | |__| | (_| | || (_| \__ \  __/ |_\__ \
        # |_____/ \__,_|\__\__,_|___/\___|\__|___/
                                                

        # self.missing_datasets = []
        self.complete_datasets = []
        self.makedataset_conditional()


        self.iterators = []
        self.create_iterators()

                                              


    def make_maskj_list(self):
        """
        Generates a mask for each dim (where the maskj for each dim
        assumes all points not on dim j are complete)
        """
        #create a maskj for each dim
        for j in range(self.m):
            masklist = []
            #construct via stacking
            for i in range(self.m):
                if i==j:
                    masklist.append(tf.gather(self.mask,j,axis=1))
                else:
                    masklist.append(tf.ones(self.n))
            maskj =tf.stack(masklist,axis=1)
            #add to list
            self.maskj_list.append(maskj)


    def create_iterators(self):
        self.iterators=[]
        for j in range(self.m):
            it = iter(self.complete_datasets[j])
            self.iterators.append(it)


    def makedataset_conditional(self):
        """
        Creates a missing and complete 
        tensorflow dataset for each dim 
        where batches always have same label
        """
        complete_datasets = []

        #dataset of all data
        initdat = tf.data.Dataset.from_tensor_slices((self.Xt,
            self.labels,self.mask))


        #complete data processing:
        for j in range(self.m):
            y0dat = initdat.filter(lambda x,y,m : CondPred(x,y,m,0,j,1))
            y1dat = initdat.filter(lambda x,y,m : CondPred(x,y,m,1,j,1))

            y0dat = y0dat.repeat().shuffle(self.n).batch(self.batch_size)
            y1dat = y1dat.repeat().shuffle(self.n).batch(self.batch_size)

            #sample equally from the two labels
            ydat = tf.data.experimental.sample_from_datasets([y0dat, y1dat], [0.5, 0.5])
            #store the dataset
            complete_datasets.append(ydat)   

        #update the datasets:
  
        self.complete_datasets = complete_datasets       


    def update_Xt(self,new_Xt):
        """
        Updates Xt in the frame
        """
        #update Xt
        self.Xt=new_Xt
        #update complete datasets
        self.makedataset_conditional()
        #reset iterators
        self.create_iterators()
        #update Xt without j'th axis
        self.Xt_no_j_list = []
        for j in range(self.m):
            Xt_no_j = tf.gather(self.Xt,self.notj_list[j],axis=1)
            self.Xt_no_j_list.append(Xt_no_j)
                         






    #  _______        _       _             
    # |__   __|      (_)     (_)            
    #    | |_ __ __ _ _ _ __  _ _ __   __ _ 
    #    | | '__/ _` | | '_ \| | '_ \ / _` |
    #    | | | | (_| | | | | | | | | | (_| |
    #    |_|_|  \__,_|_|_| |_|_|_| |_|\__, |
    #                                  __/ |
    #                                 |___/ 




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




    def impute(self,X,j):
        """
        Impute dimension j of a tensor X
        
        Parameters
        ----------
        X: tensor l x m
        j: int in range 0 to m-1

        Output
        ----------
        X_hat : tensor l x m

        """

        #gather input for imputer
        X_no_j = tf.gather(X,self.notj_list[j],axis=1)
        #predict for all values
        X_pred = self.imputers[j](X_no_j)
        #only impute missing values on dim j 
        X_hat = X*self.maskj_list[j] + X_pred*(1-self.maskj_list[j])


        return X_hat


    def imputeXt_getbatch(self,j,label):
        """
        Impute dimension j of a tensor self.Xt
        and return a batch of size getbatch
        
        Parameters
        ----------
        j: int in range 0 to m-1
        label : int (0 or 1 )

        Output
        ----------
        bm : tensor (batch_size,m) of missing vals with label from Xt

        """


        #Impute Xt using dimensions that are not j 
        X_pred = self.imputers[j](self.Xt_no_j_list[j])
        #obtain prediction
        X_hat = self.Xt*self.maskj_list[j] + X_pred*(1-self.maskj_list[j])
        #sample ids to create batch
        sample_ids = sample_without_replacement(self.mjy_ids_list[label][j],
            self.batch_size)
        bm = tf.gather(X_hat,sample_ids,axis=0)


        return bm 



    # @tf.function
    def train_imputer_step(self,bc,j,label):
        """
        Single train step for imputer j on a dataset (X,Y)
        conditioned on a label 

        Parameters
        ----------
        bc : tensor (batch_size, m) complete valued batch
        bm : tensor (batch_size, m) missing valued batch 
        j : int (dimension for which the relevant imputer will be played)
        """
        with tf.GradientTape() as tape:
            tape.watch(self.imputers[j].trainable_weights)

            #sample bm
            bm = self.imputeXt_getbatch(j,label)


            #run sinkhorn on the batches
            loss = sinkhorn(bc.shape[0],bm.shape[0],bc,bm,self.p,
                div=True,niter=self.niter,epsilon=self.eps)

            # self.losshist.append(loss.numpy())
        #perform gradient step on NN for dim j 
        tf.debugging.Assert(loss>0,[loss])
        gradients = tape.gradient(loss ,self.imputers[j].trainable_weights)

        gradients = check_gradients(gradients)
        self.opt.apply_gradients(zip(gradients,self.imputers[j].trainable_weights))

        # self.gradhist.append(gradients)



    # def forward(self,batch,wbatch1,wbatch2):
    #     """
    #     Forward of learning batch + wasserstein distance of two batches
    #     batch = tuple (data,labels)
    #     """

    #     #unpack labels for ERM batch and wasserstein batch
    #     X,Y = batch
    #     b1,b1_labels = wbatch1
    #     b2,b2_labels = wbatch2 

    #     #Predict and score ERM 
    #     out = self.classifier(X)
    #     er_loss = self.classifier_loss(out,Y)

    #     #wasserstein regularisation
    #     wloss = sinkhorn(b1.shape[0],b2.shape[0],b1,b2,self.p,div=True,niter=self.niter,epsilon=self.eps)

    #     #overall loss = er_loss + wreg*wloss
    #     loss = er_loss + self.wass_reg*wloss

    #     return loss 







