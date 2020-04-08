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

parser = argparse.ArgumentParser(description = "Plotting 3d variables")
parser.add_argument("foldername", help = "folder name of variables",type = str)
parser.add_argument("filename" , help = "File name .pickle", type = str)
parser.parse_args()
args = parser.parse_args()

path = os.path.join("./variables",args.foldername,args.filename)

with open(path,'rb') as f:

    history,data,mids,cids= pickle.load(f)
K=100
raw = history[0][::K]
print("animating ",len(raw),"iterations")

Xt = np.concatenate(raw,axis=0)
n_points = raw[0].shape[0]
T=len(raw)



t = np.array([np.ones(n_points)*i for i in range(T)]).flatten()

df = pd.DataFrame({"time": t ,"x" : Xt[:,0], "y" : Xt[:,1], "z" : Xt[:,2]})

def update_graph(num):
    data=df[df['time']==num]
    dx=np.array(data.x)
    dy=np.array(data.y)
    dz=np.array(data.z)
    scatters[0]._offsets3d = (dx[cids], dy[cids], dz[cids])
    # scatters[1]._offsets3d = (dx[mids], dy[mids], dz[mids])
    for i in range(len(mids)):
    	mid=mids[i]
    	scatters[i+1]._offsets3d = ([dx[mid]],[dy[mid]],[dz[mid]])
    title.set_text('3D Test, time={}'.format(num*K))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')



data=df[df['time']==0]
dx=np.array(data.x)
dy=np.array(data.y)
dz=np.array(data.z)
scatters = []
scatters.append(ax.scatter(dx[cids], dy[cids], dz[cids],alpha=0.4))
# scatters.append(ax.scatter(dx[mids], dy[mids], dz[mids],alpha=0.7,cmap='cool'))
for mid in mids:
	scatters.append(ax.scatter([dx[mid]],[dy[mid]],[dz[mid]]))

ani = animation.FuncAnimation(fig, update_graph, T, 
                               interval=1, blit=False)



Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
ani.save("test"+'.mp4', writer=writer)

plt.show()