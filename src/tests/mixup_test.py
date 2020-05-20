"""
File: mixup_test
Description: Test file for mixup function
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""
import numpy as np
import tensorflow as tf
from src.mixup import *
np.random.seed(123)

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


def test_mixup1d_dimensions():
	b1 = np.random.normal(size=(2,8),loc=0.0,scale=1)
	b2 = np.random.normal(size=(2,8),loc=1,scale=1)

	sup1 = [[0.],[0.1,0.3,0.8]]
	sup2 = [[1.3,2.7],[0.]]
	K=5

	X,Y =mixup_ot1d(b1,b2,sup1,sup2,K=K)

	assert X.shape==b1.shape
	assert np.shape(Y)==(X.shape[0],K)


		