"""
File: graphing 
Description: Contains functions for loading saved variables and plotting
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License
"""

import numpy as np 
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D

def loadplt3D(Xinit,Ximp,mids,cids,name):
	"""
	Loads and plots data from 3D_toy_example:

	Parameters
	-------
	Xinit : array (n x d) Initial data
	Ximp :  array (n x d) Imputed data
	mids : int list indicies of missing values
	cids : int list of complete values
	name : str 
	"""

	#load the data which has no missing values
	xinit=Xinit[cids,0]
	yinit=Xinit[cids,1]
	zinit=Xinit[cids,2]
	#load the data that has been imputed 
	xfill=Ximp[mids,0]
	yfill=Ximp[mids,1]
	zfill=Ximp[mids,2]

	#plot the 3d graph
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xinit,yinit,zinit,alpha = 0.7,color='b',marker = '.')
	ax.scatter(xfill,yfill,zfill,alpha=0.8,color ='g',marker= 'x')
	# ax.set_title("Sinkhorn Imputation %i epochs %f missing"%(epochs,per))
	ax.set_title(name)

	plt.show()

