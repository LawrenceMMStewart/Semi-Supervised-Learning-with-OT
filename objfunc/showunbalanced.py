
"""
File: showvar
Description: This file calls the animation functions found in utils to examine the
                gradient flows of 3 gaussians, one with higher variance

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
from utils.animate import *
parser = argparse.ArgumentParser(description = "Plotting 3d variables")
parser.add_argument("filename" , help = "File name .pickle", type = str)
parser.parse_args()
args = parser.parse_args()

foldername = "unbalanced"
path = os.path.join("./variables",foldername,args.filename)

with open(path,'rb') as f:

    history= pickle.load(f)
K=100
framelist = history[0][0][::K]


time = 15 #time in seconds
fps = max(len(framelist) // time ,1) #calculate fps required for a 10 second video 


c1 = [i for i in range(350)]
c2 = [i for i in range(350,700)]
c3 = [i for i in range(700,1500)]

clusterids=[c1,c2,c3]
animate3DFlow_Mixture(framelist, args.filename, clusterids,v1=[4,19] ,v2=[33,57],
 K = K, save = True , fps = fps , show = True)
