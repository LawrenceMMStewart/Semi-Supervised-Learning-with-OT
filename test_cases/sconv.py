"""
File: sconv
Description: A test to experiement how the ground cost power affects the convergance of the sinkhorn divergance

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""



import numpy as np 
import matplotlib.pyplot as plt 
from utils.datasets import *
from utils.graphing import * 
from utils.utils import *
from utils.sinkhorn import *
import tensorflow as tf
from tqdm import tqdm
from tensorflow import keras
import pickle 
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(123)


#Torus Parameters
R1,r1 = (50,1.5)
R2,r2 = (20,1.5)
n_sq = 50

var = 5

#create Tori Dataset
Tori1 = NoisyTorus(R1,R2,r1,r2,n_sq,n_sq,var)


data1 = Tori1.data
labels1 = Tori1.labels 
mframe = MissingData(data1,labels=labels1)
mframe.MCAR_Mask(0.04) 
X_start = mframe.Initialise_Nans(eta=1)

middle = len(X_start)//2

# scatter3D([X_start[:middle],X_start[middle:]])
ids = [i for i in range(len(X_start))]

def sample():
	return sample_without_replacement(ids,100).numpy()

b1=sample()
b2=sample()

bn=len(b1)
plist = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]
nlist = [i*10 for i in range(1,15)]

exp = True


if exp:

	losses = [ [] for i in range(len(plist))]

	for niter in nlist:
		for j in range(len(plist)):
			p=plist[j]
			losses[j].append(sinkhorn(bn,bn,X_start[b1],X_start[b2],p,True,niter=niter,epsilon=0.01).numpy())
	with open("./test_cases/losses.pickle",'wb') as f:
		pickle.dump([losses],f)
else:
	file = open("test_cases/losses.pickle",'rb')
	losses=pickle.load(file)[0]

for i in range(len(losses)):
	plt.plot(nlist,losses[i],label=r"$p=%.2f$"%plist[i],marker='.',alpha=0.7)

plt.title('Ground Cost Power')
plt.grid('on')
ax = plt.gca()
ax.set_facecolor('#D9E6E8')
plt.xlabel("niter")
plt.ylabel("Sinkhorn Div")
plt.legend()
plt.show()