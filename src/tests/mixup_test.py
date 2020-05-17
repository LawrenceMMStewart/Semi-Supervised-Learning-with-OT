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
