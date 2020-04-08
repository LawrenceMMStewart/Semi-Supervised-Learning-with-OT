
"""
File: showflow
Description: This file calls the animation functions found in utils to examine the
                gradient flows of two different initialisations of 

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
parser.add_argument("foldername", help = "folder name of variables",type = str)
parser.add_argument("filename" , help = "File name .pickle", type = str)
parser.parse_args()
args = parser.parse_args()

path = os.path.join("./variables",args.foldername,args.filename)

with open(path,'rb') as f:

    history,data,mids,cids= pickle.load(f)
K=100
framelist = history[0][::K]
framelist2 = history[1][::K]

breakpoint()

time = 10 #time in seconds
fps = max(len(framelist) // time ,1)

animate3DFlow_MultiView(framelist, mids, cids, "standard", K = K, save = True, fps = fps )
animate3DFlow_MultiView(framelist2, mids, cids, "mixture", K = K, save = True, fps = fps )
