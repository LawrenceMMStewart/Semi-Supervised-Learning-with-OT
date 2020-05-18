"""
File: mixup
Description: Contains functions for mixup using OT
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from ot.ot1d import *


def sample_mixup_param(n,alpha=0.75):

	#sample from beta distribution with params alpha,alpha
	l_sample = np.random.beta(alpha,alpha,
		size=[n,1])
	#take max(lambda,1-lambda)
	lambda_ops = np.concatenate((l_sample,1-l_sample),axis=1)
	lambdas = np.max(lambda_ops,axis=1).reshape(-1,1)

	return lambdas


def mixup_ot1d(batch1,batch2,sups1,sups2,alpha=0.75,K=5):
	"""
	batch1 n x m 
	batch2 n x m
	sup1 list of size n consisting of sups of batch 1 
	y2  list of size n consisting of sups of batch 2
	alpha float beta dist param
	K int number of points for barycentre
	"""

	n = batch1.shape[0]
	lambdas = sample_mixup_param(n,alpha=alpha)
	
	#generate linear combination of the batches of points
	X = batch1*lambdas+(1-lambdas)*batch2

	#create an array (lambdas,1-lambdas)
	weights = np.concatenate((lambdas,1-lambdas),axis=1)
	
	#combine supports into lists
	pointwise_join = lambda x,y: [x,y]
	xlist = list(map(pointwise_join,sups1,sups2))
	
	#create uniform point masses for each measure
	uniform_weighting = lambda in_list : np.ones(len(in_list))/len(in_list)
	uni1 = list(map(uniform_weighting,sups1))
	uni2 = list(map(uniform_weighting,sups2))

	#combine weights into lists
	µlist = list(map(pointwise_join,uni1,uni2))

	Ks = [K for i in range(n)]
	#return list of barycentres:
	Y = list(map(uniform_barycentre_1D,xlist,µlist,Ks,weights))

	return X,Y



if __name__ =="__main__":

	b1 = np.ones((5,2))
	b2 = np.zeros((5,2))

	sup1 = np.ones((5,1))
	sup2 = np.zeros((5,1))
	K=1

	X,Y =mixup_ot1d(b1,b2,sup1,sup2,K=K)
	print(X)
	print(Y)


