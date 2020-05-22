#tests for ot1d.py

import pytest
from ot.ot1d import *
import numpy as np

def precision_eq(a,b):
	return (np.abs(a-b)<1e-14).all()

def test_ttu1d_5k():
	#tests transform to uniform 1d
	µ1 = np.array([0.2,0.5,0.3])
	k=5
	answer = np.array([[0.2,0.,0.,0.,0.],
		[0.,0.2,0.2,0.1,0.],
		[0.,0.,0.,0.1,0.2]])
	guess = transport_to_uniform_1D(µ1,k)

	assert precision_eq(guess,answer)

def test_ttu1d_1k():
	µ1 = np.array([0.2,0.5,0.3])
	k=1
	answer = np.array([[0.2],
		[0.5],
		[0.3]])
	guess = transport_to_uniform_1D(µ1,k) 
	assert precision_eq(guess,answer)

def test_ttu1d_2k():
	µ1 = np.array([0.2,0.5,0.3])
	k=2
	answer = np.array([[0.2,0],
		[0.3,0.2],
		[0,0.3]])
	guess = transport_to_uniform_1D(µ1,k)
	assert precision_eq(guess,answer)	
	
	
def test_bary1d_222():
	µ1= np.array([0.8,0.2])
	x1 = np.array([1,2])

	µ2 = np.array([0.4,0.6])
	x2 = np.array([4,6])

	k=1
	answer = np.array([3.2])
	guess = uniform_barycentre_1D([x1,x2],[µ1,µ2],k)


	assert precision_eq(answer,guess)


def test_bary1d_222_rev():
	µ1= np.array([0.8,0.2])
	x1 = np.array([1,2])

	µ2 = np.array([0.4,0.6])
	x2 = np.array([4,6])

	k=1
	answer = np.array([3.2])
	guess = uniform_barycentre_1D([x2,x1],[µ2,µ1],k)

	assert precision_eq(answer,guess)



def test_bary1d_225():
	µ1= np.array([0.8,0.2])
	x1 = np.array([1,2])

	µ2 = np.array([0.4,0.6])
	x2 = np.array([4,6])

	k=5
	answer = np.array([2.5,2.5,3.5,3.5,4])
	guess = uniform_barycentre_1D([x1,x2],[µ1,µ2],k)

	assert precision_eq(answer,guess)



def test_bary1d_diracs():
	x1 = np.array([1.])
	x2 = np.array([2.])
	µ=np.array([1.])
	k=1

	answer = np.array([1.5])
	guess = uniform_barycentre_1D([x1,x2],[µ,µ],k)
	assert precision_eq(answer,guess)


def test_bary1d_diracs_weighted():
	x1 = np.array([1.])
	x2 = np.array([2.])
	µ=np.array([1.])
	k=1
	weights = [0.9,0.1]

	answer = np.array([1.10])
	guess = uniform_barycentre_1D([x1,x2],[µ,µ],k,weights=weights)
	assert precision_eq(answer,guess)


def test_bary1d_mixedsizes():
	x1 =[5.]
	x2= [2.5,2.5,2.5]
	K=1
	µ1 = [1.]
	µ2 = np.ones(len(x2))/len(x2)

	answer = np.array([3.75])
	guess = uniform_barycentre_1D([x1,x2],[µ1,µ2],K)
	assert precision_eq(answer,guess)

def test_bary1d_mixed_sizes2():
	x1 =[1.0,1]
	x2= [2.,2.0]
	K=1
	µ1 = [0.5,0.5]
	answer = np.array([1.5])
	guess = uniform_barycentre_1D([x1,x2],[µ1,µ1],K)
	assert precision_eq(answer,guess)

def test_bary1d_mixed_sizes3():
	x1 =[1.0,1]
	x2= [2.,2.0]
	K=1
	µ1 = [0.5,0.5]
	µ2 = np.ones(len(x2))/len(x2)
	answer = np.array([1.5])
	guess = uniform_barycentre_1D([x1,x2],[µ1,µ2],K)
	assert precision_eq(answer,guess)


def test_bary1d_mixed_sizes2_weighted():
	x1 =[1.0,1]
	x2= [2.,2.0]
	K=1
	µ1 = [0.5,0.5]
	weights = [0.9,0.1]
	answer = np.array([1.1])
	guess = uniform_barycentre_1D([x1,x2],[µ1,µ1],K,weights=weights)
	assert precision_eq(answer,guess)

#bary(ax,ay) = a bary(x,y)
def test_bary1d_scalarmultiplication():
	x1 = [3.0,6.0]
	x2 = [-3.1,0.5]
	K = 10
	µ = [0.2,0.8]
	weights = [0.5,0.5]

	sx1 = [a*2 for a in x1]
	sx2 = [a*2 for a in x2]
	guess = uniform_barycentre_1D([sx1,sx2],[µ,µ],K,weights=weights)
	guess_scaled = uniform_barycentre_1D([x1,x2],[µ,µ],K,weights=weights)*2

	assert precision_eq(guess,guess_scaled)

#bary(x+a,y+a) = a+bary
def test_bary1d_scalaraddition():
	x1 = [3.0,6.0]
	x2 = [-3.1,0.5]
	K = 10
	µ = [0.2,0.8]
	weights = [0.5,0.5]

	ax1 = [a+2 for a in x1]
	ax2 = [a+2 for a in x2]
	guess = uniform_barycentre_1D([ax1,ax2],[µ,µ],K,weights=weights)
	guess_scaled = uniform_barycentre_1D([x1,x2],[µ,µ],K,weights=weights)+2

	assert precision_eq(guess,guess_scaled)

