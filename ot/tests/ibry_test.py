#tests for ibpr.py


from ot.ibpr import *
import numpy as np
from ot.ot1d import cost_mat

def precision_eq(a,b):
	return (np.abs(a-b)<1e-14).all()

#test iterated bregman projections between two diracs
def test_IBP_ex1():
	# X = {0,4}
	# hists = {dirac(0),dirac(4)}
	hists = np.array([[1.,0],[0,1.]])
	cost = 4*(1-np.eye(2))
	cost = cost/np.median(cost)
	#one would expect the barycentre to be at 2 
	out = IBP(hists,cost,eps=2/hists.shape[0])
	assert precision_eq(out,np.ones((2,1))*0.5)
	assert abs(out.sum()-1)<1e-14


def test_IBP_isproba():
	#X is randomly sampled 
	d= 100
	N = 10
	#create a random ground space
	X = np.random.normal(size=(d,1))
	C = cost_mat(X,X,100,100,p=2).numpy()
	#create random historgrams 
	hists = np.random.uniform(size=(d,N))
	hists = hists/hists.sum(axis=0)

	out = IBP(hists,C,eps=2/hists.shape[0],niter=1000)
	print(out)
	assert np.shape(out)==(d,1)
	assert abs(out.sum()-1)<1e-6




