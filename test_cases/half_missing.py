"""
Test: Check that imputation of a simple 2d dataset works:
the first half of the dataset is complete and the second half of the dataset
is missing one of its two dimensional values.
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

n_samples=300

#generate the data
noisy_circles, _ = datasets.make_circles(n_samples=n_samples,factor=0.5,noise=.05)
noisy_moons, _ = datasets.make_moons(n_samples=n_samples,noise=.05)
noisy_scurve,_ = datasets.make_s_curve(n_samples=n_samples)
#make 3d data 2d
noisy_scurve=np.delete(noisy_scurve,1,1)


def missing_2nd_half(in_data):
    n=len(in_data)
    assert n%2 ==0

    obs_data=in_data.copy()
    mask=np.ones(obs_data.shape)

    #randomly replace on of the two values for each data point
    #in the second half of the dataset
    for i in range(n//2,n):
        ind=np.random.binomial(1,0.5)
        mask[i,ind]=0
        obs_data[i,ind]=np.nan

    return obs_data,mask

dataset=np.concatenate((noisy_scurve,noisy_scurve)).astype(np.float32)


data,mask = missing_2nd_half(dataset)
m=700

print("percentage of datapoints with missing values ",percentage_missing(mask),"%")
print("percentage of empty points ",percentage_empty(mask),"%")
mids,cids=generate_labels(mask)


#replace nans by means + noise 
X=Initialise_Nans(data,eta=0.2)

opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.01)
# opt=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=.01)

Xlk=tf.Variable(X)
loss_fun = lambda: sinkhorn_sq_batch(Xlk,p=2,niter=10,div=True,epsilon=0.1) #old


#training
epochs=10000


for e in tqdm(range(epochs),desc="RMSProp Maxits"):
    #compute gradient
    grads_and_vars = opt.compute_gradients(loss_fun)

    #mask the gradient (i.e. so we are calculating w.r.t to X_impute)
    mskgrads_and_vars=[(g*(1-mask),v) for g,v in grads_and_vars]
    #apply gradient step
    opt.apply_gradients(mskgrads_and_vars)



predicted_data=Xlk.numpy()
imputed_data = np.nan_to_num(data*mask)+predicted_data*(1-mask)


print(dataset[0:3,0])
print(data[0:3,0])

fig, axes = plt.subplots(1,2)
axes[0].scatter(dataset[:,0],dataset[:,1],alpha=0.8,marker='.')
axes[0].set_title("ground truth")
axes[1].scatter(data[cids,0],data[cids,1],alpha=0.7,label="observable",marker='.')
axes[1].scatter(imputed_data[mids,0],imputed_data[mids,1],alpha=0.7,label="imputed",color="g",marker='.')
axes[1].legend()
axes[1].set_title("imputation")
plt.show()



#  #to fix -- create a numpy.arraysplit for tensorflow for the sinkhorn algorithm