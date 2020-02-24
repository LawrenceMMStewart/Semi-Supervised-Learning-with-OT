import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from utils.utils import *
from utils.sinkhorn import *

np.random.seed(123)

n_samples=1500

#generate the data
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples,noise=.05)
noisy_scurve,_ = datasets.make_s_curve(n_samples=n_samples)
#for the s curve he has taken the 0th and 2nd dimension and left out the middle one
# plt.scatter(noisy_circles[:,0],noisy_circles[:,1])
# plt.show()
# plt.scatter(noisy_moons[:,0],noisy_moons[:,1])
# plt.show()
# plt.scatter(noisy_scurve[:,0],noisy_scurve[:,2])
# plt.show()

#create the observable dataset and mask:
data,mask=MCAR_Mask(noisy_circles,0.1)

#batchsize
m=100
#shuffle data set and partition into subsets of size 2m 
data=shuffle(data)
k=int(n_samples/m)

#replace nans by means + noise 
X=Initialise_Nans(data)
partitions=np.split(X,k)

#for each batch
for p in partitions:
	#seperate into Xk and Xl 
	with tf.GradientTape() as t:

		Xlk=tf.constant(p)
		t.watch(Xlk)

		#run sinkhorn
		Xk,Xl=tf.split(Xlk,2)
		l=sinkhorn_divergance(Xk,Xl,euclidean_sqdist)
		print("sinkhorn divergance is",l)
		print("computing gradient")
		grad_Xlk=t.gradient(l,Xlk)
		print(grad_Xlk)
		break








