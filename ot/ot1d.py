"""
File: ot1d
Description: 1 Dimensional OT functions

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""


import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt


@tf.function
def optimal_assignment_1D(X,Y,p):
	"""
	Returns the optimal assignement cost in 1D
	for two uniform measures consisting of n support.
	
	Parameters
	----------
	X: tensor (n,1)
	Y: tensor (n,1)
	p: int (power of euclidean cost matrix)

	Returns
	---------
	oac : float tensor (optimal assignment cost)

	"""
	X=tf.sort(X,axis=0)
	Y=tf.sort(Y,axis=0)
	oac = tf.reduce_sum(tf.abs(X-Y)**p)
	return oac


def transport_to_uniform_1D(µ,k):
	"""
	Returns the transport map for a convex cost function problem
	from a 1d measure µ into a uniform measure consisting of 
	a k point support (w.l.o.g we assume the points are sorted
	in size)

	Parameters
	----------
	µ : n x 1 array summing to 1 
	k : int 

	Output
	---------
	T : array n x k 

	"""
	n = len(µ)
	curr_loc = 0
	curr_cap = 1/k
	T = np.zeros([n,k])

	for i in range(n):

		earth = µ[i]
		while earth>1e-15: #not zero as reccurring can 
							#cause problems

			#case 1) can push all earth
			if curr_cap >earth:
				T[i][curr_loc]+= earth
				curr_cap -= earth
				earth = 0
			#case 2) equal amount 
			if curr_cap == earth:
				T[i][curr_loc]+=earth
				earth=0
				curr_cap=1/k
				curr_loc+=1
			#case 3) too much earth
			if earth>curr_cap:
				T[i][curr_loc]+=curr_cap
				earth-=curr_cap
				curr_cap=1/k
				curr_loc+=1
	return T


def uniform_barycentre_1D(xlist,µlist,K,weights=None):
	"""
	Calculates the uniform K point approximation to a barycentre
	of the measures described by xlist (supports of measures) and 
	µlist (weighting of points). An optional array weights
	may be given to control which part of the convex hull of 
	the points is returned. 
	
	Parameters
	-----------
	xlist: list (arrays of size n_i) support
	µlist: list (arrays of size n_i)measures
	k : int size of uniform approximation
	weights : array (len(xlist)) barycentre weights


	Outputs
	--------
	support : array of size k
	"""
	no_measures=len(xlist)
	support = np.zeros(K)

	#if no weights given assume weighting over barycentre
	if weights is None:
		weights = np.ones([no_measures])/no_measures

	#obtain all transport maps
	tmaps = [transport_to_uniform_1D(µ,K) for µ in µlist]

	#weight the transport maps accordingly (n =normalised)
	ntmaps = [tmaps[i]*weights[i] for i in range(no_measures)]
	C = sum([x.sum(axis=0) for x in ntmaps])
	ntmaps = [t/C for t in ntmaps]


	for k in range(K):
		#calculate the support points
		#recall support[k] = sum_i=1^n w_i* (T_i[:,k].T @ x_i)
		for i in range(no_measures):
			p = ntmaps[i][:,k]
			x = xlist[i] #obtain support of measure i 

			support[k]+=np.sum(p*x) #weighted contribution from xi -> sup[k]
			
	return support







#an example 
if __name__=="__main__":

	#test case on uniform distributions
	x1 = np.sort(np.random.uniform(1,2,100))
	x2 = np.sort(np.random.uniform(0,1,100))
	print(x1)
	print(x2)

	µ1 = np.ones(len(x1))/len(x1)
	µ2 = np.ones(len(x2))/len(x2)


	plt.figure()
	ks= [1,5,20,50,100]
	heights = [3,4,5,6,7]
	for i in range(len(ks)):
		k =ks[i]
		h=heights[i]
		ys= uniform_barycentre_1D([x1,x2],[µ1,µ2],k)
		plt.scatter(ys,[h for i in range(len(ys))],label="barycentre K=%i"%k,
			alpha=0.7)
	plt.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom=False,      # ticks along the bottom edge are off
	    top=False,         # ticks along the top edge are off
	    labelbottom=False) # labels along the bottom edge are off
	plt.scatter(x1,[1 for i in range(len(x1))],label="uniform support [0,1]",alpha=0.7)
	plt.scatter(x2,[2 for i in range(len(x2))],label="uniform support [1,2]",alpha=0.7)
	plt.legend()
	plt.grid('on')
	ax = plt.gca()
	ax.set_facecolor('#D9E6E8')
	plt.title("Barycentres for Uniform Measures with Supports Sampled from Uniform Distributions")
	plt.show()