from multiprocessing import cpu_count

import numpy as np
import tensorflow as tf


@tf.function
def augment(x):
    """
    augments a data point x by flipping and cropping images
    """
    # random left right flipping
    x = tf.image.random_flip_left_right(x)
    # random pad and crop
    x = tf.pad(x, paddings=[(0, 0), (4, 4), (4, 4), (0, 0)], mode='REFLECT')
    x = tf.map_fn(lambda batch: tf.image.random_crop(batch, size=(32, 32, 3)), x, parallel_iterations=cpu_count())
    return x


def guess_labels(u_aug, model, K):
    """
    takes the mean prediction of a list of augmented 
    datapoints (of size K) and returns the mean of the logits.
    """
    u_logits = tf.nn.softmax(model(u_aug[0]), axis=1)
    for k in range(1, K):
        u_logits = u_logits + tf.nn.softmax(model(u_aug[k]), axis=1)
    u_logits = u_logits / K
    u_logits = tf.stop_gradient(u_logits)
    return u_logits


@tf.function
def sharpen(p, T):
    """
    sharpens logits p using parameter T
    """
    return tf.pow(p, 1/T) / tf.reduce_sum(tf.pow(p, 1/T), axis=1, keepdims=True)


@tf.function
def mixup(x1, x2, y1, y2, beta):
    beta = tf.maximum(beta, 1-beta)
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    return x, y


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [tf.concat(v, axis=0) for v in xy]


def mixmatch(model, x, y, u, T, K, beta):
    batch_size = x.shape[0]
    x_aug = augment(x)
    u_aug = [None for _ in range(K)]
    for k in range(K):
        u_aug[k] = augment(u)
    mean_logits = guess_labels(u_aug, model, K)
    qb = sharpen(mean_logits, tf.constant(T))
    U = tf.concat(u_aug, axis=0)
    qb = tf.concat([qb for _ in range(K)], axis=0)
    XU = tf.concat([x_aug, U], axis=0)
    XUy = tf.concat([y, qb], axis=0)
    indices = tf.random.shuffle(tf.range(XU.shape[0]))
    W = tf.gather(XU, indices)
    Wy = tf.gather(XUy, indices)
    XU, XUy = mixup(XU, W, XUy, Wy, beta=beta)
    XU = tf.split(XU, K + 1, axis=0)
    XU = interleave(XU, batch_size)
    return XU, XUy


#we want list of size batch size [2,32 x 32 x 3]

def mixmatch_ot(model, x, y, u, T, K, beta,
    M,
    eps=None,niter=200,tol=1e-5):

    batch_size = x.shape[0]
    x_aug = augment(x)
    #u_aug will be a list of size K 
    #where each element is of size [batch_size,32,32,3] for cifar10
    u_aug = [None for _ in range(K)]
    q_aug = [None for _ in range(K)]
    for k in range(K):
        u_aug[k] = augment(u)
        q_aug[k] = tf.stop_gradient(tf.nn.softmax(model(u_aug[k]), axis=1))
    q_aug = [q[:,:,None] for q in q_aug]
    q_aug = np.concatenate(q_aug,axis=2)

    #run Iterative bregman projections on each example
    mapping_fn  = lambda x,M=M,eps=eps,niter=niter,tol=tol: IBP(x,
        M,eps=eps,niter=niter,tol=tol).flatten()
    qb = list(map(mapping_fn,q_aug))
    #concatenate list to origional array form
    qb = np.vstack(qb)

    # #old --
    # mean_logits = guess_labels(u_aug, model, K)
    # qb = sharpen(mean_logits, tf.constant(T))
    #------

    U = tf.concat(u_aug, axis=0)
    qb = tf.concat([qb for _ in range(K)], axis=0)
    XU = tf.concat([x_aug, U], axis=0)
    XUy = tf.concat([y, qb], axis=0)
    indices = tf.random.shuffle(tf.range(XU.shape[0]))
    W = tf.gather(XU, indices)
    Wy = tf.gather(XUy, indices)
    XU, XUy = mixup(XU, W, XUy, Wy, beta=beta)
    XU = tf.split(XU, K + 1, axis=0)
    XU = interleave(XU, batch_size)
    return XU, XUy


@tf.function
def semi_loss(labels_x, logits_x, labels_u, logits_u):
    loss_xe = tf.nn.softmax_cross_entropy_with_logits(labels=labels_x, logits=logits_x)
    loss_xe = tf.reduce_mean(loss_xe)
    loss_l2u = tf.square(labels_u - tf.nn.softmax(logits_u))
    loss_l2u = tf.reduce_mean(loss_l2u)
    return loss_xe, loss_l2u


def linear_rampup(epoch, rampup_length=16):
    if rampup_length == 0:
        return 1.
    else:
        rampup = np.clip(epoch / rampup_length, 0., 1.)
        return float(rampup)


def weight_decay(model, decay_rate):
    for var in model.trainable_variables:
        var.assign(var * (1 - decay_rate))

#exponential moving
def ema(model, ema_model, ema_decay):
    for var, ema_var in zip(model.variables, ema_model.variables):
        if var.trainable:
            ema_var.assign((1 - ema_decay) * var + ema_decay * ema_var)
        else:
            ema_var.assign(tf.identity(var))


def IBP(hists,M,eps=None,weights=None,niter=100,tol=1e-6):
    """
    Calculates the barycentre of a set of histograms using
    the iterative bregman projection approach detailed in the paper
    Benamou et al. 

    Parameters
    -----------
    hists : array (d x n) [n histograms of coming from the d simplex]  
    M : array (d x d) [ground metric, please normalise with median of metric]
    eps : float [regularisation parameter 2/d is a natural choice] 
    weights : array [if None then set as isometric by default]
    niter : int [maximum number of iterations to run IBP]
    tol : float [tolerance for convergance]
    """

    d = hists.shape[0] #dimension of hists [i.e size of space]
    n = hists.shape[1] #number of hists

    if eps is None:
        eps = 2/d

    #define the kernel
    K = np.exp(-M/eps)
    #numerical trick seen in gpeyre implementation
    K[K<1e-300]=1e-300

    counter = 0
    diff = np.inf 

    if weights is None:
        weights = np.ones(n)/n

    #initialise u0 v0 as ones
    v,u = (np.ones((d,n)),np.ones((d,n)))
    uKv = u*(K@v)
    #weighted log of uKv
    wloguKv = weights*np.log(uKv)

    #prod uKv^weights = exp( sum {weights*log(uKv)})
    bary  = np.exp(wloguKv.sum(axis=1)).reshape(-1,1)
    prev_bary = np.copy(bary)

    for i in range(1,niter):
        #update v 
        v = hists / (K.T@u)
        #update barycentre
        uKv = u*(K@v)
        wloguKv = weights*np.log(uKv)
        #prod uKv**weights = exp( sum weights*log(uKv))
        bary  = np.exp(wloguKv.sum(axis=1)).reshape(-1,1)
        if i%10 ==0:
            if np.sum(bary-prev_bary)<tol:
                break
        prev_bary =np.copy(bary)
        #update u 
        u = bary / (K@v)

    return bary
