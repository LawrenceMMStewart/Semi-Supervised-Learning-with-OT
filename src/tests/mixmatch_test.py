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

#test mixmatch 
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
	assert (Uprime>=np.tile(u,[3,1])).all()
	assert Xprime.shape == x.shape
	assert Uprime.shape[0] == 3*u.shape[0]
	assert Uprime.shape[1]==u.shape[1]

	assert len(Yprime)==len(y)
	assert len(Qprime)==3*len(y)
	assert len(Yprime[0])==30
	assert len(Qprime[0])==30 



# #test mixmatchloss_1d (MSE loss is functional)
# def test_mixmatchloss_1d():
# 	Y=  np.array([[1.],[2.]]).astype(np.float32)
# 	Yhat=  np.array([[1.],[1.]]).astype(np.float32)
# 	#mse(Y,Yhat)  = 0.5
# 	Q =  np.array([[0.],[2.]]).astype(np.float32)
# 	Qhat =  np.array([[0.],[1.]]).astype(np.float32)
# 	#mse(Q,Qhat) = 0.5 
# 	reg=tf.constant(0.1,dtype=tf.float32)
# 	guessx,guessu = mixmatchloss_1d(Y,Yhat,Q,Qhat)
# 	guess = guessx+reg*guessu

# 	answer = tf.constant(0.5)+0.1*tf.constant(0.5)
# 	assert guess==answer

#mix match loss (x,x) (q,q) = 0
def test_mixmatchloss_ot_d1():
	y1 = np.array([[1.0],[1.1],[1.3]]).astype(np.float32)
	y2 = np.array([[3.0],[2.95],[3.1]]).astype(np.float32)
	Y = np.array([y1,y2])
	

	q1 = np.array([[2.0],[2.01],[1.987]]).astype(np.float32)
	q2 = np.array([[5.1],[5.13],[5.09]]).astype(np.float32)
	Q = np.array([q1,q2])


	lx, lu = mixmatchloss_ot(Y,Y,Q,Q)

	assert lx == 0  
	assert lu == 0 


#mix match loss (x,y) = (y,x) (q,p) = (p,q)
def test_mixmatchloss_ot_d2():
	y1 = np.array([[1.0],[1.1],[1.3]]).astype(np.float32)
	y2 = np.array([[3.0],[2.95],[3.1]]).astype(np.float32)
	Y = np.array([y1,y2])
	

	q1 = np.array([[2.0],[2.01],[1.987]]).astype(np.float32)
	q2 = np.array([[5.1],[5.13],[5.09]]).astype(np.float32)
	Q = np.array([q1,q2])

	Qhat = tf.constant([ [[2.15]],[[5.76]] ],dtype=tf.float32)
	Yhat = tf.constant([[[1.0]],[[3.0]]],dtype=tf.float32)

	lx1, lu1 = mixmatchloss_ot(Y,Yhat,Q,Qhat)
	lx2, lu2 = mixmatchloss_ot(Yhat,Y,Qhat,Q)

	assert lx1 == lx2 
	assert lu1 == lu2


def test_mixmatchloss_ot_exact():
	x = tf.constant([[0.]],dtype=tf.float32)
	y = tf.constant([[-0.5],[0.5]],dtype=tf.float32)

	Y = [x,x]
	Yhat = [y,y]

	q = tf.ones((9,1),dtype=tf.float32)
	q2 = tf.zeros((6,1),dtype=tf.float32)
	Q = [q,q2]

	Qhat = [q,q2]

	lx,lu = mixmatchloss_ot(Y,Yhat,Q,Qhat)
	assert lx ==0.25
	assert lu ==0

	