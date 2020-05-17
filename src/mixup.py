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


def mixup_ot1d(batch1,batch2,y1,y2,alpha=0.75):
	"""
	batch1 n x m 
	batch2 n x m
	y1 n x m
	y2 n x m 
	alpha float beta dist param
	"""

	l_sample = np.random.beta(alpha,alpha,
		size=[batch1.shape[0],1])
	#take max(lambda,1-lambda)
	lambda_ops = np.concatenate((l_sample,1-l_sample),axis=1)
	lambdas = np.max(lambda_ops,axis=1).reshape(-1,1)

	return lambdas




