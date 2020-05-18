"""
File: mixup_test
Description: Test file for mixup function
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
import numpy as np
import tensorflow as tf
from src.mixup import *

def test_lambdasample():
	n = 100
	sample = sample_mixup_param(n,alpha=0.75)
	assert sample.shape == (n,1)
	assert (sample>=0.5).all()


def test_mixup1d_diraclabels_K1():
	b1 = np.ones((5,2))
	b2 = np.zeros((5,2))

	sup1 = np.ones((5,1))
	sup2 = np.zeros((5,1))
	K=1

	X,Y =mixup_ot1d(b1,b2,sup1,sup2,K=K)

	for i in range(len(X)):
		assert X[i][0]==Y[i][0]
		