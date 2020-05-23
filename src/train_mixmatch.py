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
import seaborn as sns
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
parser.add_argument("n_labels",
    help = "Number of labels to train on [4000?,2000,1000,500,250]",
    type = int)
parser.add_argument("device",
    help="options = [GPU:x,CPU:0]")
args = parser.parse_args()

#define the device
dev = "/"+args.device
#if GPU lock to single device:
if dev != "/CPU:0":
    os.environ["CUDA_VISIBLE_DEVICES"]=dev[-1]



def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print("Available GPUS:",get_available_gpus())


dname = args.dataset
n_labels = args.n_labels

#save losses to tensorboard
run_name = str(n_labels) + "-"+datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join("src","logs",dname,"mixmatch",run_name)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#train and validation directories
tdir = os.path.join(logdir,"train")
vdir = os.path.join(logdir,"validation")
#train and validation summary writers
tsw = tf.summary.create_file_writer(tdir)
vsw = tf.summary.create_file_writer(vdir)



#initialisations for dataset
scaler = MinMaxScaler()  
if dname =="wine":
    path = os.path.join("datasets","wine","winequality-white.csv")
data = pd.read_csv(path,sep=';')
X = data.drop(columns='quality')
Y = data['quality']
#fit the scaler to X 
scaler.fit(X)

#split into train and test sets
train_x,test_x,train_y,test_y = train_test_split(X,Y,
    random_state = 0, stratify = Y,shuffle=True,
    train_size=4000)

batch_size = 32

#note that when batching there maybe excess data with uneven batches

#create test set 
test   = scaler.transform(test_x)
test_y  = pd.DataFrame.to_numpy(test_y).reshape(-1,1).astype(np.float32)

#labelled train set
trainX  = scaler.transform(train_x[:n_labels])
train_y  = pd.DataFrame.to_numpy(train_y[:n_labels]).reshape(-1,1)

#unlabelled train set
trainU  = scaler.transform(train_x[:n_labels])

#which dataset is smaller



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

#mixmatch OT loss for training
loss_fn = lambda Y,Yhat,Q,Qhat,reg: mixmatchloss_ot1d(Y,
    Yhat,Q,Qhat,reg=reg,
    niter=tf.constant(50),
    epsilon=tf.constant(0.1,dtype=tf.float32),
    p=tf.constant(1,dtype=tf.float32))

#function to evaluate performance on validation set (MSE)
def evaluate(val_metric = tf.keras.losses.MSE):
    pred = model(test)
    losses = val_metric(pred,test_y)
    mloss = tf.reduce_sum(losses)
    return mloss

#training
opt = tf.keras.optimizers.Adam()
# epochs = 25000
epochs = 20

for e in tqdm(range(epochs),desc="Epoch"):
    for step,batch in enumerate(data_labelled):

        Xbatch = batch[0].numpy()
        Ybatch = batch[1].numpy()
        Ubatch = next(it).numpy()
        #to prevent uneven sized batches in final batch
        Ubatch = Ubatch[:len(Xbatch)] 


        #perform mixmatch
        Xprime,Yprime,Uprime,Qprime = mixmatch_ot1d(model,Xbatch,
            Ybatch,Ubatch,stddev=0.01,alpha=0.75,K=1,naug=3)
        breakpoint()

        Qprime = tf.expand_dims(Qprime,axis=2)
        Yprime = tf.expand_dims(Yprime,axis=2)
        
        Qprime = tf.cast(Qprime,tf.float32)
        Yprime = tf.cast(Yprime,tf.float32)
        reg = tf.constant(10.0,dtype = tf.float32)

        #on first epoch lossx approx 5 lossu approx 0.5
        with tf.GradientTape() as tape:

            predx = model(Xprime)
            predu = model(Uprime)

            #expand dimensions for loss function in parallel
            predx = tf.expand_dims(predx,axis=2)
            predu = tf.expand_dims(predu,axis=2)

            loss = loss_fn(Yprime,predx,Qprime,predu,reg)

        #calculate gradients and update
        gradients = tape.gradient(loss ,model.trainable_weights)
        opt.apply_gradients(zip(gradients,model.trainable_weights))
        print("done step 2 ")
    #write train and validation losses (Wasserstein/ MSE r.) to tensorboard
    with tsw.as_default():
        tf.summary.scalar('train loss', loss, step=e)
    val_loss = evaluate()
    with vsw.as_default():
        tf.summary.scalar('val error',val_loss, step=e)
    

#save model
save_path = os.path.join("src","models",dname,
    "mixmatch",str(n_labels))
model.save(save_path)



