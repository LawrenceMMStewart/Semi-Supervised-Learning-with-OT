"""
File: tori-roundrobin
Description: A test for the imputers utils file to ensure that imputation using the round robin 

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from utils.datasets import *
from utils.graphing import *
from utils.imputers import * 
from utils.utils import *
from utils.sinkhorn import *
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
import pickle 
from mpl_toolkits.mplot3d import Axes3D
import os
np.random.seed(123)
# tf.random.set_seed(123)



parser = argparse.ArgumentParser(description = "Sinkhorn Batch Imputation for 3D Dataset")
parser.add_argument("batch_size",help = "size of each batch for imputation",type = int) 
parser.add_argument("T",help = "Number of round robin loops", type = int)
parser.add_argument("epochs",help = "training iterations per robin", type = int)
parser.add_argument("niter",help = "number of iterations of sinkhorn",type =int)
parser.parse_args()
args = parser.parse_args()



gpuid = '0'
gpu = False
#set to whichever device is free
os.environ["CUDA_VISIBLE_DEVICES"]=gpuid

if gpu:
    dev = "/GPU:"+gpuid
else:
    dev = "/CPU:0"
print("Available GPUS:",get_available_gpus())

# with tf.device(dev):

#Torus Parameters
R1,r1 = (9,0.5)
R2,r2 = (3,0.5)
n_sq = 50
var = 0.5
Tori = NoisyTorus(R1,R2,r1,r2,n_sq,n_sq,var)
data = Tori.data
labels = Tori.labels 

#insert nans as missing data
mframe = MissingData(data,labels=labels)

# create the observable dataset and mask:
mframe.MCAR_Mask(0.04) #0.04
per = mframe.percentage_missing()
print("Percentage of missing data = ",per)

#mids = positions of points that have missing values, cids = positions of points that are complete
mids,cids = mframe.Generate_ids()
#initialise nans as samples from normal dist
X_start = mframe.Initialise_Nans(eta=1)


Xt = tf.constant(X_start,dtype =tf.float32)
labels = tf.constant(labels,dtype=tf.float32)
mask = tf.constant(mframe.mask,dtype = tf.float32)

networks = ClassifyImpute(Xt,labels,mask,niter=args.niter,
    optimiser =tf.keras.optimizers.Adam(),p=1.0) #learning_rate=1e-5 for adam previously


#training for T epochs
for t in tqdm(range(args.T),desc = "RR Iter:"):
    #for each imputer
    for j in tqdm(range(mframe.m),desc = "Imputer"):
        #train impute j for epochs iterations conditioning on label
        for i in tqdm(range(args.epochs),desc="Epoch"):

            #sample a batch of complete ids
            bc,lbc,_ = next(networks.iterators[j])
            #sample a batch from the missing data 
            #conditional on the label of bc
            cond_label = int(lbc[0][0].numpy())
            with tf.device(dev):
                networks.train_imputer_step(bc,j,cond_label)
        #after training imputer j update Xt
        Xt = networks.impute(Xt,j)
        networks.update_Xt(Xt)


Xft1 = Xt.numpy()[networks.labels_ids_list[0].numpy()]
Xft2 = Xt.numpy()[networks.labels_ids_list[1].numpy()]

scatter3D([Xft1,Xft2],colorlist =['b','g'],
    markerlist=['.','.'],alphalist=[0.4,0.4]) 