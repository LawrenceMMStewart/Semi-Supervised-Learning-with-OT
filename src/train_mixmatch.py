"""
File: train_mixmatch 
Description: Train MLP in semi-supervised setting using mixmatch (ot 1d approach)
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import os
import argparse 
from src.mixmatch import *
from ot.sinkhorn import * 
from tqdm import tqdm
#seed the RNG 
np.random.seed(123)
tf.random.set_seed(123)

#args = number of labels to train on
parser = argparse.ArgumentParser(description = "Training Arguements-")
parser.add_argument("dataset",
    help="Options = wine,")
parser.add_argument("device",
    help="options = [GPU:x,CPU:0]")
parser.add_argument("n_labels",
    help = "Number of labels to train on",
    type = int)
parser.add_argument("batch_size",
    help = "batch size",
    type = int )
parser.add_argument("max_reg",
    help = "maximum value regularisation parameter for loss term Lu",
    type=float)
parser.add_argument("K",
    help = "Number of points for uniform approximation of Barycentre",
    type = int)
parser.add_argument("noise_amp",
    help = "noise amplitude / stddev for mixmatch augmentations",
    type=float)
parser.add_argument("naug",
    help = "number of augmentations for mixmatch",
    type = int)
args = parser.parse_args()


dev = "/"+args.device


def create_name(varlist,taglist):
    """
    returns file name e.g. [10,2] ['n','m']
    ---> n10-m2
    """
    assert len(varlist)==len(taglist)
    run_tag = taglist[0]+str(varlist[0])
    for i in range(1,len(varlist)):
        run_tag+="-"+taglist[i]+str(varlist[i])
    return run_tag


with tf.device(dev):

    dname = args.dataset
    n_labels = args.n_labels
    batch_size = args.batch_size

    run_tag = create_name([args.n_labels,args.batch_size,args.max_reg,
        args.noise_amp,args.K,args.naug],
        ["n",'b','r','amp','K','aug'])


    # save losses to tensorboard
    run_name = run_tag + "-"+datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("src","logs",dname,"mixmatch","ot",run_name)


    #tensorboard paths 
    trainloss_path = os.path.join(logdir,"trainloss")
    mse_path = os.path.join(logdir,"mse")
    consistancy_path = os.path.join(logdir,"consistancy") 
    validation_path = os.path.join(logdir,"validation")
    regulariser_path = os.path.join(logdir,"regulariser")
    #tensorboard writers 
    trainloss_w = tf.summary.create_file_writer(trainloss_path)
    mse_w = tf.summary.create_file_writer(mse_path)
    consistancy_w = tf.summary.create_file_writer(consistancy_path)
    validation_w = tf.summary.create_file_writer(validation_path)
    regulariser_w = tf.summary.create_file_writer(regulariser_path)



    if dname == "wine":
        train,test,train_y,test_y = load_wine()


    #only use an user-chosen amount of data for training
    train = train[:n_labels]
    train_y = train_y[:n_labels]

    #labelled train set
    trainX  = train[:n_labels]
    train_y  = train_y[:n_labels]

    #unlabelled train set
    trainU  = train[n_labels:]


    #define model
    l2reg=1e-3
    d=trainX.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2*d, activation ='relu',input_shape = (d,),
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
        tf.keras.layers.Dense(d,activation = 'relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
        tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(l2reg))
        ])


    model.summary()

    #create tf.datasets for X and U 
    nx = trainX.shape[0]
    nu = trainU.shape[0]

    #labelled data batched
    data_labelled = tf.data.Dataset.from_tensor_slices((trainX.astype(np.float32),
        train_y.astype(np.float32)))
    data_labelled = data_labelled.shuffle(nx).batch(batch_size)

    #unlimited random stream of unlabelled data
    data_unlabelled = tf.data.Dataset.from_tensor_slices((trainU.astype(np.float32)))
    data_unlabelled = data_unlabelled.shuffle(nu).repeat().batch(batch_size)
    it = iter(data_unlabelled)


    #function to evaluate performance on validation set (MSE)
    def evaluate(val_metric = tf.keras.losses.MSE):
        pred = model(test)
        losses = val_metric(pred,test_y)
        mloss = tf.reduce_mean(losses)
        return mloss

    #training
    opt = tf.keras.optimizers.Adam()
    epochs = 25000
    
    #ramp up regularisation parameter throughout time 
    #maxing out at 16000 iterations (as in mixmatch paper)
    reach_max = 16000


    regsramp = np.linspace(0,args.max_reg,num=reach_max)
    regsflat = np.ones(epochs-reach_max)*args.max_reg
    regs = np.concatenate((regsramp,regsflat))

    for e in tqdm(range(epochs),desc="Epoch"):
        for step,batch in enumerate(data_labelled):

            Xbatch = batch[0].numpy()
            Ybatch = batch[1].numpy()
            Ubatch = next(it).numpy()
            #to prevent uneven sized batches in final batch
            Ubatch = Ubatch[:len(Xbatch)] 

            #perform mixmatch
            Xprime,Yprime,Uprime,Qprime = mixmatch_ot1d(model,Xbatch,
                Ybatch,Ubatch,
                stddev=args.noise_amp,alpha=0.75,K=args.K,naug=args.naug)



            reg = tf.constant(regs[e],dtype = tf.float32)

            #on first epoch lossx approx 5 lossu approx 0.5
            with tf.GradientTape() as tape:

                predx = tf.expand_dims(model(Xprime),2)
                predu = tf.expand_dims(model(Uprime),2)

                lossx,lossu = mixmatchloss_1d(Yprime,predx,
                    Qprime,predu)
                loss = lossx+reg*lossu

            #calculate gradients and update
            gradients = tape.gradient(loss ,model.trainable_weights)
            opt.apply_gradients(zip(gradients,model.trainable_weights))


        #write losses and regularises to tensorboard
        with trainloss_w.as_default():
            tf.summary.scalar('Loss', loss, step=e)
        with mse_w.as_default():
            tf.summary.scalar('Lx',lossx, step=e)
        with consistancy_w.as_default():
            tf.summary.scalar('Lu',lossu, step=e)
        with regulariser_w.as_default():
            tf.summary.scalar("Regulariser",reg,step=e)
        val_loss = evaluate()
        with validation_w.as_default():
            tf.summary.scalar('MSE Validation',val_loss, step=e)
        

    #save model
    save_path = os.path.join("src","models",dname,
        "mixmatch","ot",run_tag)
    model.save(save_path)



