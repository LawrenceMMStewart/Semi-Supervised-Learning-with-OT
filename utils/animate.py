"""
File: animate
Description: Contains functions for animating (used in the gradient descent)
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License
"""

import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def update3D(iteration, data, mids,scatters):
    """
    Update the data held by the scatter plot and therefore animates it.

    Parameters:
    ------------
    iteration (int): Current iteration of the animation
    data (list): List of the data positions at each iteration.
    mids (list): indicies of data points that have missing values
    scatters (list): List of all the scatters (One per element)

    Returns:
    ------------
    list: List of scatters (One per element) with new coordinates
    """

    #plot the complete data
    for i in range(len(mids)):
    	mid = mids[i]
    	#i+1 as first entry is for the cids 
        scatters[i+1]._offsets3d = (data[iteration][mid,0:1], data[iteration][mid,1:2], 
        	data[iteration][mid,2:],alpha = 0.8)
    return scatters

def animate3D(data,mids,cids, save=False,title="Gradient Flow"):
    """
    Creates the 3D figure and animates it with the input data.
    
    Parameters
    ------------
    data (list): List of the data positions at each iteration.
    save (bool): Whether to save the recording of the animation. (Default to False).
    """

    # Attaching 3D axis to the figure


    #Initialize scatters 1 plot for all the cids and 1 plot for each of the mids
    scatters= []
    scatters.append(ax.scatter(data[0][cids,0],data[0][cids,1],data[0][cids,2],alpha=0.6))
    for mid in mids:
    	scatters.append(ax.scatter(data[0][mid,0],data[0][mid,1],data[0][mid,2],alpha =0.8))

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-2, 2])
    ax.set_xlabel('X')

    ax.set_ylim3d([-2, 2])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-2, 2])
    ax.set_zlabel('Z')

    ax.set_title(title)

    # Provide starting angle for the view.
    ax.view_init(25, 10)

    ani = animation.FuncAnimation(fig, update3D, iterations, fargs=(data, mids,scatters),
                                       interval=50, blit=False, repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save(title+'.mp4', writer=writer)

    plt.show()
