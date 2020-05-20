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

def mixmatch_ot1d_dimtest():

	#data X  = ones  (3,5)
	#f  =  sum over the columns
	#U = 0.5 (3,5)
	model = lambda x : tf.reshape(tf.reduce_sum(x,axis=1),(-1,1))
	x = np.ones((3,5)) 
	y = np.ones((3,1))*5
	u  = np.ones((3,5))*0.5
	Xprime,Yprime,Uprime,Qprime = mixmatch_ot1d(model,x,y,u,
		stddev=0.0,K=30,naug=3)

	assert (Xprime< x).all()
	assert (Uprime>u).all()
	assert Xprime.shape == x.shape
	assert Uprime.shape == u.shape
	assert Yprime.shape == (3,30)
	assert Qprime.shape == (3,30)

