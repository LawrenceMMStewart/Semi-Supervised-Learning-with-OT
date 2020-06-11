"""
Iterative bregman projections for barycentre
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Liscence: Mit License
"""
from functools import reduce
import numpy as np




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





if __name__ =="__main__":
	#two point distribution 
	
	#3 hists over two points
	hists =  np.array([[0.25,0.25,0.25,0.25],[0.5,0.4,0.1,0.0]]).T
	print("histograms cols = hists",hists)
	cost = (1-np.eye(4))

	out = IBP_Barycentre(hists,cost,eps= 2/hists.shape[0],niter=1000)
	print("sum of bary",out.sum())
	print(out)

# def gab_bary(hists,M,eps=0.1,weights=None,niter=10):
# 	"""
# 	hists : d x n [n histograms of size d]  
# 	M : ground metric [expected to normalise via using median] d x d
# 	eps : regulariser 
# 	"""

# 	n = hists.shape[1] #number of hists
# 	d = hists.shape[0] #dimension of hists [i.e size of space]

# 	#define the kernel
# 	K = np.exp(-M/eps)
# 	#numerical trick seen in gpeyre implementation
# 	K[K<1e-300]=1e-300

# 	counter = 0
# 	diff = np.inf 

# 	if weights is None:
# 		weights = np.ones(n)/n

# 	#two first projections

# 	uKv = K@( (hists.T/K.sum(axis=0)).T )
# 	u = np.exp(weights*np.log(uKv))/uKv

# 	for i in range(niter):

# 		uKv = u*(K@hists/(K.T@u))
# 		u = u*np.exp(weights*np.log(uKv))/uKv

# 	bary = uKv.mean(axis=1)

# 	return bary

