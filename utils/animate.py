"""
File: animate
Description: Contains functions for animating (used to show gradient flow)
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License
"""

from utils.animate import * 
import matplotlib.pyplot as plt
import pickle
import argparse
import os.path
import numpy as np
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import pandas as pd



def animate3DFlow(framelist,mids,cids,name,K=1,save=False,fps=30):
    """
    Animates a 3d animation of gradient flow

    Parameters
    ----------
    framelist : list of n x d arrays (frames)
    mids : int list 
    cids : int list
    name : str 
    K    : Scaling factor for title e.g. if one wishes to take
           a frame list that consists of every K'th iterations
    save : Boolean
    fps  : int  
    """

    print("animating ",len(framelist),"iterations")
    Xt = np.concatenate(framelist,axis=0)
    n_points = framelist[0].shape[0]
    T=len(framelist)

    t = np.array([np.ones(n_points)*i for i in range(T)]).flatten()
    df = pd.DataFrame({"time": t ,"x" : Xt[:,0], "y" : Xt[:,1], "z" : Xt[:,2]})


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title(r'Gradient Flow' )
    # plt.gca().patch.set_facecolor('white')


    data=df[df['time']==0]
    dx=np.array(data.x)
    dy=np.array(data.y)
    dz=np.array(data.z)
    scatters = []
    scatters.append(ax.scatter(dx[cids], dy[cids], dz[cids],alpha=0.2))
    # scatters.append(ax.scatter(dx[mids], dy[mids], dz[mids],alpha=0.7,cmap='cool'))
    for mid in mids:
        scatters.append(ax.scatter([dx[mid]],[dy[mid]],[dz[mid]],marker='.',alpha =0.8))
    ax.view_init(elev=7,azim=88)

    def update_graph(num):
        data=df[df['time']==num]
        dx=np.array(data.x)
        dy=np.array(data.y)
        dz=np.array(data.z)
        scatters[0]._offsets3d = (dx[cids], dy[cids], dz[cids])
        for i in range(len(mids)):
            mid=mids[i]
            scatters[i+1]._offsets3d = ([dx[mid]],[dy[mid]],[dz[mid]])
        title.set_text(r'Gradient Flow, $t={}$'.format(num*K))


    ani = animation.FuncAnimation(fig, update_graph, T,interval=1, blit=False)


    if save:

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(name+'.mp4', writer=writer)

    plt.show()





def animate3DFlow_MultiView(framelist,mids,cids,name,v1=[9,73],
    v2=[0,90],K=1,save=False,fps=30,show=True):
    """
    Animates a 3d animation of gradient flow
    with 2 plots at different views 

    Parameters
    ----------
    framelist : list of n x d arrays (frames)
    mids : int list 
    cids : int list
    name : str 
    v1  : float list of length 2 elev and azim of plot 1
    v2  : float list of length 2 elev and azim of plot 2
    K    : Scaling factor for title e.g. if one wishes to take
           a frame list that consists of every K'th iterations
    save : Boolean
    fps  : int  
    """

    print("Animating ",len(framelist),"iterations")
    Xt = np.concatenate(framelist,axis=0)
    n_points = framelist[0].shape[0]
    T=len(framelist)

    t = np.array([np.ones(n_points)*i for i in range(T)]).flatten()
    df = pd.DataFrame({"time": t ,"x" : Xt[:,0], "y" : Xt[:,1], "z" : Xt[:,2]})


    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    title = plt.suptitle(r'Gradient Flow')



    data=df[df['time']==0]
    dx=np.array(data.x)
    dy=np.array(data.y)
    dz=np.array(data.z)
    scatters1 = []
    scatters2 = []

    scatters1.append(ax1.scatter(dx[cids], dy[cids], dz[cids],alpha=0.2))
    scatters2.append(ax2.scatter(dx[cids], dy[cids], dz[cids],alpha=0.2))

    for mid in mids:
        scatters1.append(ax1.scatter([dx[mid]],[dy[mid]],[dz[mid]],marker='x',alpha =0.8))
        scatters2.append(ax2.scatter([dx[mid]],[dy[mid]],[dz[mid]],marker='x',alpha =0.8))

    ax1.view_init(elev=v1[0],azim=v1[1])
    ax2.view_init(elev=v2[0],azim=v2[1])



    def update_graph(num):
        data=df[df['time']==num]
        dx=np.array(data.x)
        dy=np.array(data.y)
        dz=np.array(data.z)
        scatters1[0]._offsets3d = (dx[cids], dy[cids], dz[cids])
        scatters2[0]._offsets3d = (dx[cids], dy[cids], dz[cids])

        for i in range(len(mids)):
            mid=mids[i]
            scatters1[i+1]._offsets3d = ([dx[mid]],[dy[mid]],[dz[mid]])
            scatters2[i+1]._offsets3d = ([dx[mid]],[dy[mid]],[dz[mid]])
    
        title.set_text(r'Gradient Flow, $t={}$'.format(num*K))


    ani = animation.FuncAnimation(fig, update_graph, T,interval=1, blit=False)


    plt.tight_layout()
    if save:

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(name+'.mp4', writer=writer)
    if show:
        plt.show()


def animate3DFlow_Mixture(framelist,name,clusterids,v1=[9,73],
    v2=[0,90],K=1,save=False,fps=30,show=True):
    """
    Animates a 3d animation of gradient flow
    with 2 plots at different views 

    Parameters
    ----------
    framelist : list of n x d arrays (frames)
    name : str 
    clusterids : list of lists each list is the indicies for a cluster
    v1  : float list of length 2 elev and azim of plot 1
    v2  : float list of length 2 elev and azim of plot 2
    K    : Scaling factor for title e.g. if one wishes to take
           a frame list that consists of every K'th iterations
    save : Boolean
    fps  : int  
    """

    print("Animating ",len(framelist),"iterations")
    Xt = np.concatenate(framelist,axis=0)
    n_points = framelist[0].shape[0]

    T=len(framelist)

    t = np.array([np.ones(n_points)*i for i in range(T)]).flatten()
    df = pd.DataFrame({"time": t ,"x" : Xt[:,0], "y" : Xt[:,1], "z" : Xt[:,2]})


    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    title = plt.suptitle(r'Gradient Flow')



    data=df[df['time']==0]
    dx=np.array(data.x)
    dy=np.array(data.y)
    dz=np.array(data.z)
    scatters =[[] for i in range(len(clusterids))]

    for i in range(len(clusterids)):
        cluster = clusterids[i]
        scatters[i].append(ax1.scatter(dx[cluster], dy[cluster], dz[cluster],alpha=0.6,marker = '.'))
        scatters[i].append(ax2.scatter(dx[cluster], dy[cluster], dz[cluster],alpha=0.6,marker = '.'))


    ax1.view_init(elev=v1[0],azim=v1[1])
    ax2.view_init(elev=v2[0],azim=v2[1])

  



    def update_graph(num):
        data=df[df['time']==num]
        dx=np.array(data.x)
        dy=np.array(data.y)
        dz=np.array(data.z)

        for i in range(len(clusterids)):
            cluster = clusterids[i]

            scatters[i][0]._offsets3d = (dx[cluster], dy[cluster], dz[cluster])
            scatters[i][1]._offsets3d = (dx[cluster], dy[cluster], dz[cluster])

       
  
    
        title.set_text(r'Gradient Flow, $t={}$'.format(num*K))


    ani = animation.FuncAnimation(fig, update_graph, T,interval=1, blit=False)


    plt.tight_layout()
    if save:

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(name+'.mp4', writer=writer)
    if show:
        plt.show()














def animate3DFlow_dualgauss(framelist,name,v1=[9,73],
    v2=[0,90],K=1,save=False,fps=30,show=True):
    """
    Animates a 3d animation of gradient flow
    with 2 plots at different views 

    Parameters
    ----------
    framelist : list of n x d arrays (frames)
    name : str 
    v1  : float list of length 2 elev and azim of plot 1
    v2  : float list of length 2 elev and azim of plot 2
    K    : Scaling factor for title e.g. if one wishes to take
           a frame list that consists of every K'th iterations
    save : Boolean
    fps  : int  
    """

    print("Animating ",len(framelist),"iterations")
    Xt = np.concatenate(framelist,axis=0)
    n_points = framelist[0].shape[0]
    c1 = [i for i in range(n_points//2)] #cluster 1 
    c2 = [i for i in range(n_points//2,n_points)] #cluster 2
    T=len(framelist)

    t = np.array([np.ones(n_points)*i for i in range(T)]).flatten()
    df = pd.DataFrame({"time": t ,"x" : Xt[:,0], "y" : Xt[:,1], "z" : Xt[:,2]})


    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    title = plt.suptitle(r'Gradient Flow')



    data=df[df['time']==0]
    dx=np.array(data.x)
    dy=np.array(data.y)
    dz=np.array(data.z)
    scatters1 = []
    scatters2 = []

    scatters1.append(ax1.scatter(dx[c1], dy[c1], dz[c1],alpha=0.6,marker = '.',color='g'))
    scatters2.append(ax2.scatter(dx[c1], dy[c1], dz[c1],alpha=0.6,marker = '.',color='g'))


    scatters1.append(ax1.scatter(dx[c2],dy[c2],dz[c2],marker='.',alpha =0.6,color='b'))
    scatters2.append(ax2.scatter(dx[c2],dy[c2],dz[c2],marker='.',alpha =0.6,color='b'))

    ax1.view_init(elev=v1[0],azim=v1[1])
    ax2.view_init(elev=v2[0],azim=v2[1])



    def update_graph(num):
        data=df[df['time']==num]
        dx=np.array(data.x)
        dy=np.array(data.y)
        dz=np.array(data.z)

        scatters1[0]._offsets3d = (dx[c1], dy[c1], dz[c1])
        scatters2[0]._offsets3d = (dx[c1], dy[c1], dz[c1])

       
            
        scatters1[1]._offsets3d = (dx[c2],dy[c2],dz[c2])
        scatters2[1]._offsets3d = (dx[c2],dy[c2],dz[c2])
    
        title.set_text(r'Gradient Flow, $t={}$'.format(num*K))


    ani = animation.FuncAnimation(fig, update_graph, T,interval=1, blit=False)


    plt.tight_layout()
    if save:

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(name+'.mp4', writer=writer)
    if show:
        plt.show()