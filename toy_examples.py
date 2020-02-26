"""
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Liscence: Mit License 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from test_cases.utils import *
import tensorflow_probability as tfp
from tqdm import tqdm

np.random.seed(123)	

n_samples=500 #1500

#generate the data
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples,noise=.05)
noisy_scurve,_ = datasets.make_s_curve(n_samples=n_samples)
#make 3d data 2d
noisy_scurve=np.delete(noisy_scurve,1,1)


#for the s curve he has taken the 0th and 2nd dimension and left out the middle one
# plt.scatter(noisy_circles[:,0],noisy_circles[:,1])
# plt.show()
# plt.scatter(noisy_moons[:,0],noisy_moons[:,1])
# plt.show()
# plt.scatter(noisy_scurve[:,0],noisy_scurve[:,1])
# plt.show()

# create the observable dataset and mask:
data,mask=MCAR_Mask(noisy_circles,0.10) #0.15 for the other 2 

#batchsize
m=100
#shuffle data set and partition into subsets of size 2m 
data,mask=shuffle(data,mask)

print("percentage of datapoints with missing values ",percentage_missing(mask),"%")
print("percentage of empty points ",percentage_empty(mask),"%")
mids,cids=generate_labels(mask)


k=int(data.shape[0]/m)

#replace nans by means + noise 
X=Initialise_Nans(data)
#split the data into partitions

partitions=np.array_split(X,k)

#optimiser
# optimizer=tf.keras.optimizers.RMSprop()
optimizer=tf.keras.optimizers.Adam()

imputed_list=[]

#for each batch
for i in tqdm(range(len(partitions)),desc="Imputation of folds"):
	#seperate into Xk and Xl 
	Xlk=tf.Variable(partitions[i])
	loss_fun = lambda: sinkhorn_sq_batch(Xlk,niter=30)

	losses=tfp.math.minimize(loss_fun,optimizer=optimizer,num_steps=2500)
	imputed_list.append(Xlk.numpy())



predicted_data=np.concatenate(imputed_list)
imputed_data = np.nan_to_num(data*mask)+predicted_data*(1-mask)



fig, axes = plt.subplots(1,2)
axes[0].scatter(noisy_moons[:,0],noisy_moons[:,1])
axes[0].set_title("ground truth")
axes[1].scatter(data[cids,0],data[cids,1],alpha=0.8,label="observable")
axes[1].scatter(imputed_data[mids,0],imputed_data[mids,1],alpha=0.5,label="imputed",marker='x')
axes[1].legend()
axes[1].set_title("imputation")
plt.show()



 #to fix -- create a numpy.arraysplit for tensorflow for the sinkhorn algorithm