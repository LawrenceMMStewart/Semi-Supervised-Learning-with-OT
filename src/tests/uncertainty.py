"""
File: train_mixmatch 
Description: Train MLP outputting gaussians (test to see if uncertainty works)
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import os 
import tensorflow_probability as tfp

from tqdm import tqdm
#seed the RNG 
np.random.seed(123)
tf.random.set_seed(123)
tfd = tfp.distributions

#gaussian 1   (µ,s) = (2,2)   ; gaussian 2 (µ,s) = (-2,2)
g1 = tfd.Normal(2,2)
g2 = tfd.Normal(-2,2)

d=1 #if d is not equal to one use tfd.MultivariateNormal() 
#create dataset
x1 = g1.sample((300,d))
y1 = tf.ones((300,1),dtype=tf.float32)

x2 = g2.sample((300,d))
y2 = tf.zeros((300,1),dtype=tf.float32)

X = tf.concat((x1,x2),axis=0)
Y = tf.concat((y1,y2),axis=0)



#define model
l2reg=1e-3

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2*d, activation ='relu',input_shape = (d,),
        kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
    tf.keras.layers.Dense(d,activation = 'relu',
        kernel_regularizer=tf.keras.regularizers.l2(l2reg)),
    tf.keras.layers.Dense(2,kernel_regularizer=tf.keras.regularizers.l2(l2reg))
    ])

model.summary()

#shuffle and batch X
data= tf.data.Dataset.from_tensor_slices((X,Y)).batch(128)





def W2Gaussian_1D(µs1,µs2):
    """
    Calculates the wasserstein distance for 1d gaussians
    """
    return tf.reduce_sum((µs1-µs2)**2,axis=1)

def gen_gauss(params):
    return tfd.Normal(params[0],params[1])
def map_neglogprobs(a,b):
    return -a.log_prob(b)

#training
opt = tf.keras.optimizers.Adam()
epochs = 5000

for e in tqdm(range(epochs),desc="Epoch"):
    for step,batch in enumerate(data):

        Xbatch = batch[0]
        Ybatch = batch[1]


        with tf.GradientTape() as tape:
            out = model(Xbatch)
            µ,s = tf.split(out,2,axis=1)
            #clip to minimum std and maximum std
            s= tf.clip_by_value(s, clip_value_min=1e-5, clip_value_max=0.5)


            preds = tf.concat((µ,s),axis=1)


            #generate the gaussians for the predictions and labels
            pred_dists = list(map(gen_gauss,preds))
            nlp  = list(map(map_neglogprobs,pred_dists,Ybatch))
            nlp = tf.concat(nlp,axis=0)
            loss = tf.reduce_mean(nlp)


        #calculate gradients and update
        gradients = tape.gradient(loss ,model.trainable_weights)
        opt.apply_gradients(zip(gradients,model.trainable_weights))

    tqdm.write("Epoch %i ; Loss = %.3f"%(e,loss))


sb1 = tf.constant([[-2]],dtype=tf.float32)
sb0 = tf.constant([[2]],dtype=tf.float32)
print("should be 1 with high certainty ",model(sb1),"\n")
print("should be 0 with high certainty ",model(sb0),"\n")
print("should be unsure ",model(tf.zeros((1,1),dtype=tf.float32)),"\n")

breakpoint()



#----------------------------------------------------------------
#for wasserstein it was as follows:
# for e in tqdm(range(epochs),desc="Epoch"):
#     for step,batch in enumerate(data):

#         Xbatch = batch[0]
#         Ybatch = batch[1]
#         zerodevs = tf.zeros(Ybatch.shape,dtype=tf.float32)

#         with tf.GradientTape() as tape:
#             out = model(Xbatch)
#             µ = tf.gather(out,[0],axis=1)
#             s = tf.abs(tf.gather(out,[1],axis=1))

#             preds = tf.concat((µ,s),axis=1)
#             labels = tf.concat((Ybatch,zerodevs),axis=1)

#             losses = W2Gaussian_1D(preds,labels)
#             loss = tf.reduce_mean(losses)
        

#         #calculate gradients and update
#         gradients = tape.gradient(loss ,model.trainable_weights)
#         opt.apply_gradients(zip(gradients,model.trainable_weights))

#     tqdm.write("Epoch %i ; Loss = %.3f"%(e,loss))
