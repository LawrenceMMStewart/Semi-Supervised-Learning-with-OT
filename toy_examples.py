"""
Author Lawrence Stewart <lawrence.stewart@ens.fr>
Liscence: Mit License 
"""

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from utils.utils import *
from utils.sinkhorn import *
import tensorflow_probability as tfp
from tqdm import tqdm

np.random.seed(123) 

n_samples = 500 #1500

#generate the data
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples,noise=.05)
noisy_scurve,_ = datasets.make_s_curve(n_samples=n_samples)
#make 3d data 2d
noisy_scurve = np.delete(noisy_scurve,1,1)


# create the observable dataset and mask:
data,mask = MCAR_Mask(noisy_scurve,0.10) #0.15 for the other 2 

#batchsize
m = 250
#shuffle data set and partition into subsets of size 2m 
data, mask = shuffle(data,mask)

data, mask = (data.astype(np.float32), mask.astype(np.float32))

print("percentage of datapoints with missing values ",percentage_missing(mask),"%")
print("percentage of empty5points ",percentage_empty(mask),"%")

mids, cids = generate_labels(mask)


k = int(data.shape[0]/m)

#replace nans by means + noise 
X = Initialise_Nans(data)
#split the data into k partitions
data_partitions = np.array_split(X,k)
#split the masks into partitions 
mask_partitions = np.array_split(mask,k)
#optimiser
opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.01)


imputed_list = []
epochs = 50000

loss_fun = lambda: sinkhorn_sq_batch(Xlk,p=2,niter=10,div=True,epsilon=0.1)

#iterate over batches Xlk
for i in tqdm(range(len(data_partitions)),desc="Imputation of folds"):
    #seperate into Xk and Xl 
    Xlk=tf.Variable(data_partitions[i])
    #obtain the mask Xkl
    msk = mask_partitions[i]
    #update that batch 
    for e in tqdm(range(epochs),desc="RMSProp Maxits"):
        #compute gradient
        grads_and_vars = opt.compute_gradients(loss_fun)

        #mask the gradient (i.e. so we are calculating w.r.t to X_impute)
        mskgrads_and_vars=[(g*(1-msk),v) for g,v in grads_and_vars]
        #apply gradient step
        opt.apply_gradients(mskgrads_and_vars)
    imputed_list.append(Xlk.numpy())




predicted_data=np.concatenate(imputed_list)
imputed_data = np.nan_to_num(data*mask)+predicted_data*(1-mask)



fig, axes = plt.subplots(1,2)
axes[0].scatter(noisy_scurve[:,0],noisy_scurve[:,1])
axes[0].set_title("ground truth")
axes[1].scatter(data[cids,0],data[cids,1],alpha=0.8,label="observable")
axes[1].scatter(imputed_data[mids,0],imputed_data[mids,1],alpha=0.5,label="imputed",marker='x')
axes[1].legend()
axes[1].set_title("imputation")
plt.show()
    

 #to fix -- create a numpy.arraysplit for tensorflow for the sinkhorn algorithm