"""
File: test_SHGrad
Description: Test case to ensure gradient of sinkhorn is calculated correction

Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 

Consider the problem of 2 uniform point clouds, running sinkhorn w.r.t. one of them and
optimising via gradient descent we would expect them to overlap
"""

import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.utils import shuffle
from utils.sinkhorn import *  
from utils.utils import *
import tensorflow_probability as tfp
from tqdm import tqdm
import datetime
np.random.seed(123)

#initialse two point clouds on the unit grid 

#uniform taking value in 0 -> 0.4
X = np.random.uniform(0,0.4,[50,2]).astype(np.float32)
Y = np.random.uniform(0.6,1,[50,2]).astype(np.float32)


#minimise the sinkhorn distance with respect to X 
x1=tf.Variable(X)
x2=tf.Variable(X)

optimizer=tf.keras.optimizers.RMSprop()

# loss_fun = lambda : sinkhorn_divergance(x,Y,euclidean_sqdist,epsilon=0.1,niter=10)

n=X.shape[0]
m=Y.shape[0]
p=2
niter=100

epsilon=0.01


# loss_fun1 =  lambda x : sinkhorn(n,m,x,Y,p,True,niter=niter,epsilon=epsilon)

# losses=tfp.math.minimize(loss_fun,optimizer=optimizer,num_steps=1000,trainable_variables=[x])

# print("S(X,Y) = ",sinkhorn(n,m,X,Y,p,div,niter=niter,epsilon=epsilon))
# print("S(X_new,Y) = ",sinkhorn(n,m,x,Y,p,div,niter=niter,epsilon=epsilon))
# print("S(Y,Y) = ",sinkhorn(n,m,Y,Y,p,div,niter=niter,epsilon=epsilon))


# losses2 = tfp.math.minimize(loss_fun2,optimizer=optimizer,num_steps=1000,trainable_variables=[x2])
opt= tf.keras.optimizers.Adam()


#tensorboard 
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)



for t in tqdm(range(20000)):
	with tf.GradientTape() as tape:
		tf.summary.trace_on()	
		lossx1 = sinkhorn(n,m,x1,Y,p,True,niter=niter,epsilon=epsilon)
		with train_summary_writer.as_default():
			tf.summary.trace_export(name="shtrace",step=t,
	      profiler_outdir=train_log_dir)


		tf.summary.scalar('lossx1', lossx1, step=t)
	gradients1 = tape.gradient(lossx1 ,x1)

	with tf.GradientTape() as tape:
		lossx2 = sinkhorn(n,m,x2,Y,p,False,niter=niter,epsilon=epsilon)
	with train_summary_writer.as_default():

		tf.summary.scalar('lossx2', lossx2, step=t)
	gradients2 = tape.gradient(lossx2,x2)



	opt.apply_gradients(zip([gradients1],[x1]))
	opt.apply_gradients(zip([gradients2],[x2]))


plt.scatter(X[:,0],X[:,1],label="X",alpha=0.8,marker='x')
plt.scatter(Y[:,0],Y[:,1],label="Y",alpha=0.8)
plt.scatter(x1.numpy()[:,0],x1.numpy()[:,1],label="X Sinkhorn Div",alpha=0.7,marker='x')
plt.scatter(x2.numpy()[:,0],x2.numpy()[:,1],label="X regularised Transport",alpha=0.7,marker='.')
ax = plt.gca()
ax.set_facecolor('#D9E6E8')
plt.grid('on')
plt.legend()
plt.title("Transport on Uniform Point Clouds")
plt.show()





