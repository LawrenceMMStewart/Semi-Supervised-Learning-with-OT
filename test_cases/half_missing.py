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

#choose a dataset
choice = noisy_moons

#concatenate dataset 2x to see if we can retrieve missing values
data=np.concatenate((choice,choice)).astype(np.float32)

#initialise the class
mframe = MissingData(data)
obs_data, mask = mframe.missing_secondhalf2D()


print("percentage of datapoints with missing values ",mframe.percentage_missing,"%")
print("percentage of empty points ",mframe.percentage_empty(),"%")
mids,cids = mframe.Generate_Labels()


#replace nans by means + noise 
X=mframe.Initialise_Nans(eta=0.1)

opt=tf.compat.v1.train.RMSPropOptimizer(learning_rate=.01)
# opt=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=.01)

Xlk=tf.Variable(X)
loss_fun = lambda: sinkhorn_sq_batch(Xlk,p=2,niter=10,div=True,epsilon=0.1) #old


#training
epochs=500


for e in tqdm(range(epochs),desc="RMSProp Maxits"):
    #compute gradient
    grads_and_vars = opt.compute_gradients(loss_fun)

    #mask the gradient (i.e. so we are calculating w.r.t to X_impute)
    mskgrads_and_vars=[(g*(1-mask),v) for g,v in grads_and_vars]
    #apply gradient step
    opt.apply_gradients(mskgrads_and_vars)



predicted_data=Xlk.numpy()
imputed_data = np.nan_to_num(obs_data*mask)+predicted_data*(1-mask)


fig, axes = plt.subplots(1,2)
axes[0].scatter(data[:n_samples,0],data[:n_samples,1],alpha=0.8,marker='.')
axes[0].set_title("ground truth")
axes[1].scatter(obs_data[cids,0],obs_data[cids,1],alpha=0.7,label="observable",marker='.')
axes[1].scatter(imputed_data[mids,0],imputed_data[mids,1],alpha=0.7,label="imputed",color="g",marker='.')
axes[1].legend()
axes[1].set_title("imputation")
plt.show()



#  #to fix -- create a numpy.arraysplit for tensorflow for the sinkhorn algorithm