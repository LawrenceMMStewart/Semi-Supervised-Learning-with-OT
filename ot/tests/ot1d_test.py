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
