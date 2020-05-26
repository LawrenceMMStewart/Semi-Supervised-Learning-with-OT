"""
File: mixmatch_test
Description: Test file for mixup function
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""

import numpy as np
import tensorflow as tf
from src.mixmatch import * 
np.random.seed(123)

def test_mixmatch_ot1d_dim():

	#data X  = ones  (3,5)
	#f  =  sum over the columns
	#U = 0.5 (3,5)
	model = lambda x : tf.reshape(tf.reduce_sum(x,axis=1),(-1,1))
	x = np.ones((3,5)) 
	y = np.ones((3,1))*5
	u  = np.ones((3,5))*0.5
	Xprime,Yprime,Uprime,Qprime = mixmatch_ot1d(model,x,y,u,
		stddev=0.0,K=30,naug=3)
	
	assert (Xprime<=x).all() 
	assert (Uprime>=u).all()
	assert Xprime.shape == x.shape
	assert Uprime.shape == u.shape

	assert len(y)==len(Yprime)
	assert len(Qprime)==len(y)
	assert len(Yprime[0])==30
	assert len(Qprime[0])==30 



#test mixmatchloss_1d (MSE loss is functional)
def test_mixmatchloss_1d():
	Y=  np.array([[1.],[2.]]).astype(np.float32)
	Yhat=  np.array([[1.],[1.]]).astype(np.float32)
	#mse(Y,Yhat)  = 0.5
	Q =  np.array([[0.],[2.]]).astype(np.float32)
	Qhat =  np.array([[0.],[1.]]).astype(np.float32)
	#mse(Q,Qhat) = 0.5 
	reg=tf.constant(0.1,dtype=tf.float32)
	guessx,guessu = mixmatchloss_1d(Y,Yhat,Q,Qhat)
	guess = guessx+reg*guessu

	answer = tf.constant(0.5)+0.1*tf.constant(0.5)
	assert guess==answer