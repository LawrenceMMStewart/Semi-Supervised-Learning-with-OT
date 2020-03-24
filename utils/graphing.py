"""
File: graphing 
Description: Contains functions for loading saved variables and plotting
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
from mpl_toolkits.mplot3d import Axes3D

def loadplt3D(pickle_file):
	"""
	Loads and plots data from 3D_toy_example:

	Parameters
	-------
	pickle_file : String (path to pickle file)
	"""
	with open(pickle_file,'rb') as f:
		X_start ,data, imputed_data ,mids,cids = pickle.load(f)

	#load the data which has no missing values
	xinit=data[cids,0]
	yinit=data[cids,1]
	zinit=data[cids,2]
	#load the data that has been imputed 
	xfill=imputed_data[mids,0]
	yfill=imputed_data[mids,1]
	zfill=imputed_data[mids,2]

	#plot the 3d graph
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(xinit,yinit,zinit,alpha = 0.7,color='b',marker = '.')
	ax.scatter(xfill,yfill,zfill,alpha=0.8,color ='g',marker= 'x')
	# ax.set_title("Sinkhorn Imputation %i epochs %f missing"%(epochs,per))
	ax.set_title(pickle_file)

	plt.show()