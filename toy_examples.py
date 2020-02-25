import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from utils.utils import *
from utils.sinkhorn import *
import tensorflow_probability as tfp
from tqdm import tqdm

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
data,mask=MCAR_Mask(noisy_moons,0.1)
print(data,mask)

#batchsize
m=100
#shuffle data set and partition into subsets of size 2m 
data,mask=shuffle(data,mask)
k=int(n_samples/m)

print(data,mask)

# #replace nans by means + noise 
# X=Initialise_Nans(data)
# #split the data into partitions
# partitions=np.split(X,k)

# #optimiser
# optimizer=tf.keras.optimizers.RMSprop()

# imputed_list=[]

# #for each batch
# for i in tqdm(range(len(partitions)),desc="Imputation of folds"):
# 	#seperate into Xk and Xl 
# 	Xlk=tf.Variable(partitions[i])
# 	loss_fun = lambda: sinkhorn_sq_batch(Xlk,niter=50)
# 	losses=tfp.math.minimize(loss_fun,optimizer=optimizer,num_steps=3000)
# 	imputed_list.append(Xlk.numpy())

# predicted_data=np.concatenate(imputed_list)
# imputed_data = data*mask+predicted_data*(1-mask)


# plt.figure()
# plt.subplots(121)
# plt.scatter(noisy_circles[:,0],noisy_circles[:,1])
# plt.title("Ground truth")
# plt.subplots(122)
# plt.scatter(data[:,0],data[:,1],alpha=0.2)
# plt.scatter(imputed_data[:,0],imputed_data[:,1])
# plt.title("imputation")
# plt.show()
