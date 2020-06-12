"""
Iterative bregman projections for barycentre
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Liscence: Mit License
"""
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt






def IBP(hists,M,eps=0.1,weights=None,niter=100,tol=1e-13):
	"""
	Calculates the barycentre of a set of histograms using
	the iterative bregman projection approach detailed in the paper
	Benamou et al. 

	Parameters
	-----------
	hists : array (n x d) [n histograms of size d]  
	M : array (d x d) [ground metric, please normalise with median of metric]
	eps : float [regularisation parameter 2/d is a natural choice]
	weights : array [if None then set as isometric by default]
	niter : int [maximum number of iterations to run IBP]
	tol : float [tolerance for convergance]
	"""

	n = hists.shape[1] #number of hists
	d = hists.shape[0] #dimension of hists [i.e size of space]

	#define the kernel
	K = np.exp(-M/eps)
	#numerical trick seen in gpeyre implementation
	K[K<1e-300]=1e-300

	counter = 0
	diff = np.inf 

	if weights is None:
		weights = np.ones(n)/n

	#initialise u0 v0 as ones
	v,u = (np.ones((d,n)),np.ones((d,n)))
	uKv = u*(K@v)
	#weighted log of uKv
	wloguKv = weights*np.log(uKv)

	#prod uKv^weights = exp( sum {weights*log(uKv)})
	bary  = np.exp(wloguKv.sum(axis=1)).reshape(-1,1)
	prev_bary = np.copy(bary)

	for i in range(1,niter):
		#update v 
		v = hists / (K.T@u)
		#update barycentre
		uKv = u*(K@v)
		wloguKv = weights*np.log(uKv)
		#prod uKv**weights = exp( sum weights*log(uKv))
		bary  = np.exp(wloguKv.sum(axis=1)).reshape(-1,1)
		if i%10 ==0:
			if np.sum(bary-prev_bary)<tol:
				break
		prev_bary =np.copy(bary)
		#update u 
		u = bary / (K@v)

	return bary


def sharpen(d,T):
	"""
	Sharpen a histogram d with sharpness
	parameter T
	"""
	return (d**T)/(d**T).sum()

def avsharp(hists,T=0.5):
	av = hists.sum(axis=1)
	return sharpen(av,T).reshape(-1,1)





if __name__ =="__main__":
	import tensorflow as tf
	import pandas as pd
	import seaborn as sns
	def cost_mat(X,Y,n,m,p=2):
	    """
	    Returns table of pointwise Eucildean Distances
	    C_ij=|| x_i - y_j ||^2

	    Parameters
	    ----------
	    X : (tensor) (n x p) 
	    Y : (tensor) (m x p)
	    n : int
	    m : int 
	    p : int

	    Output
	    ---------
	    C : (tensor) (n x m) 
	    """
	    XX = tf.reduce_sum(tf.multiply(X,X),axis=1)
	    YY = tf.reduce_sum(tf.multiply(Y,Y),axis=1)
	    C1 = tf.transpose(tf.reshape(tf.tile(XX,[m]),[m,n]))
	    C2 = tf.reshape(tf.tile(YY,[n]),[n,m])
	    C3 = tf.transpose(tf.linalg.matmul(Y,tf.transpose(X)))
	    C = C1 + C2 - 2*C3;
	    if p == 2:
	        return C
	    else:
	        return tf.sqrt(C+10**(-12))**p



	#example 1 : simulating the softmax predictions of a neural network for 
	#			 3 noisy augmentations of a data point, comparing the sharpening
	# 			 vs barycentre approaches.

	run1 = True
	if run1:

		#ground space Z mod 10
		X = np.arange(0,5,1).reshape(-1,1)
		#generate 3 predictions
		pred1 = np.array([0.0,0.07,0.8,0.13,0]).reshape(-1,1)
		pred2 = np.array([0.05,0.05,0.7,0.15,0.05]).reshape(-1,1)
		pred3 = np.array([0.4,0.4,0.1,0,0.1]).reshape(-1,1)

		assert pred1.sum() ==1
		assert pred2.sum() ==1
		assert pred3.sum() ==1
		
		hists = np.concatenate((pred1,pred2,pred3),axis=1)
		Cost = cost_mat(X,X,5,5).numpy()
		Cost = Cost/np.median(Cost)

		bary = IBP(hists,Cost,eps = 2/hists.shape[0],niter=1000).flatten()
		mix = avsharp(hists,T=0.5).flatten()

		sns.set_style("dark")
		pred_df=pd.DataFrame({'pred1':pred1.flatten()/3, 'pred2':pred2.flatten()/3,
		 'pred3':pred3.flatten()/3})
		pred_df.plot(kind='bar', stacked=True)
		plt.xticks(rotation=30, horizontalalignment="center")
		plt.xlabel("Label")
		plt.ylabel(r"$P(Y = label)$")
		plt.show()

		bary_df = pd.DataFrame({"barycentre":bary,"Sharpen":mix})
		bary_df.plot(kind='bar')
		plt.xticks(rotation=30, horizontalalignment="center")
		plt.xlabel("Label")
		plt.ylabel(r"$P(Y = label)$")
		plt.show()


	#example 2 - comparing the barycentre of N(1,1) and N(-1,1) across
	# a discretised grid


	run2 = True
	if run2:

		sns.set_style("dark")
		def normal(x,µ,s):
			Z = (2*np.pi*(s**2))**0.5
			return np.exp((-0.5*(x-µ)**2)/s**2)/Z
		N =75
		X = np.linspace(-4,4,N)
		Cost = cost_mat(X.reshape(-1,1),X.reshape(-1,1),N,N).numpy()
		µ1 = normal(X,1,1)
		µ2 = normal(X,-1,1)
		µ1,µ2 = (µ1/µ1.sum(),µ2/µ2.sum())


		hists = np.concatenate((µ1.reshape(-1,1),µ2.reshape(-1,1)),axis=1)
		bary = IBP(hists,Cost,eps = 2/hists.shape[0],niter=1000).flatten()

		ndf=pd.DataFrame({'µ=1':µ1, 'µ=-1':µ2,"bary":bary})
		ndf.plot(kind='bar',stacked=False)
		plt.xlabel(r"$x$")
		plt.ylabel(r"$P(Y = x)$")
		plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False)
		plt.show()








