import numpy as np

#load some logits directly from the first iteration of the model
#and ensure the the IBP parameters are calibrated correctly



def test_IBP(hists,M,eps=None,weights=None,niter=100,tol=1e-6):
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
            	print(f"IBP has been terminated at iteration {i}")
            	return bary
        prev_bary =np.copy(bary)
        #update u 
        u = bary / (K@v)

    return bary



hists = np.load("tests/q_aug.npy")
M  = np.load("groundmetrics/nm-cifar10.npy")


#just for the kth index
k = np.random.randint(0,len(hists))

#give loads of its
unlim = test_IBP(hists[k],M,niter=10000,tol=1e-16)
lim  = test_IBP(hists[k],M,niter=100,tol=1e-5)

print(np.sum(lim-unlim))

