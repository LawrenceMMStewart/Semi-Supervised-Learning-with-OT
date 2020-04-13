"""
File: datasets
Description: Contains classes for custom classification tasks
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NoisyTorus():

	def __init__(self,R1,R2,r1,r2,n_sq1,n_sq2,var):
		"""
		Class for creating a noisy torus dataset, consisting of one inner and one outer torus
		used for classification.

		Parameters
		----------
		R1 : float R1 is the distance from the center of the tube to the center of the torus 1 
		R2 : float R2 is the distance from the center of the tube to the center of the torus 2 
		r1 : float r1 is the radius of the tube
		r2 : float r2 is the radius of the tube
		n_sq1 : int sqrt of number of points in torus 1 
		n_sq2 : int sqrt of number of points of torus 2 
		var : float variance for normally distributed noise

		"""
		assert R1 >r1 , "if R1 is not greater than r1, a non classical torus will be generated"
		assert R2 >r2 , "if R2 is not greater than r2, a non classical torus will be generated"
		assert (R1 >R2) , "please enter parameters for largest torus first"

		self.R1 = R1
		self.R2 = R2

		self.r1 = r1 
		self.r2 = r2 

		self.n1 = n_sq1**2
		self.n2= n_sq2**2

		self.var =var 

		self.T1 = np.concatenate(self.noisy_torus(R1,r1,n_sq1,var),axis=1)
		self.T2 = np.concatenate(self.noisy_torus(R2,r2,n_sq2,var),axis=1)

		self.data = np.concatenate((self.T1,self.T2),axis=0)
		self.labels = np.concatenate((np.ones((n_sq1**2,1)),np.zeros((n_sq2**2,1))),axis=0)

	def noisy_torus(self,R,r,n_sq,var):

		"""
		Generates a noisy torus

		Parameters
		----------
		R : float R is the distance from the center of the tube to the center of the torus  
		r : float r1 is the radius of the tube
		n_sq : int sqrt of number of points in torus
		var : float variance for normally distributed noise
			
		Outputs
		-------
		Torus : tuple (x,y,z) float n x 3 
		"""
		theta = np.linspace(0, 2.*np.pi, n_sq)
		phi = np.linspace(0, 2.*np.pi, n_sq)
		theta, phi = np.meshgrid(theta, phi)

		x = (R + r*np.cos(theta)) * np.cos(phi)
		y = (R + r*np.cos(theta)) * np.sin(phi)
		z = r * np.sin(theta)


		var=0.5
		noisex =np.random.normal(0,var,size =x.shape)
		noisey =np.random.normal(0,var,size =x.shape)
		noisez =np.random.normal(0,var,size =x.shape)
		x=(x+noisex).flatten()
		y=(y+noisey).flatten()
		z=(z+noisez).flatten()

		return (x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1))



	def show(self):
		"""
		Creates a scatter plot of the tori
		"""
		fig = plt.figure()
		ax1 = fig.add_subplot(121, projection='3d')
		ax1.set_zlim(-self.R1,self.R1)
	
		ax1.scatter(self.T1[:,0], self.T1[:,1], self.T1[:,2], color='b',alpha = 0.6)
		ax1.scatter(self.T2[:,0], self.T2[:,1], self.T2[:,2], color='g',alpha = 0.6)
		ax1.view_init(23, 27)
		ax2 = fig.add_subplot(122, projection='3d')
		ax2.set_zlim(-self.R1,self.R1)
		ax2.scatter(self.T1[:,0], self.T1[:,1], self.T1[:,2], color='b',alpha = 0.6)
		ax2.scatter(self.T2[:,0], self.T2[:,1], self.T2[:,2], color='g',alpha = 0.6)
		ax2.view_init(90, 10)
		ax2.set_xticks([])
		plt.show()




#example:
if __name__=="__main__":
		
	test =NoisyTorus(30,10,0.3,0.3,25,25,1)
	test.show()




