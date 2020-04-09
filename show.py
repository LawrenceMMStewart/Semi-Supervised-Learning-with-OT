from utils.graphing import *
import matplotlib.pyplot as plt
import pickle
import argparse
import os.path
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

parser = argparse.ArgumentParser(description = "Plotting 3d variables")
parser.add_argument("foldername", help = "folder name of variables",type = str)
parser.add_argument("filename" , help = "File name .pickle", type = str)
parser.parse_args()
args = parser.parse_args()

path = os.path.join("./variables",args.foldername,args.filename)

with open(path,'rb') as f:

	X_start, data, Xt, mids, cids, mframe = pickle.load(f)

per = "%.3f %%"%mframe.percentage_missing()

name = per+" missing data"

loadplt3D(X_start,Xt,mids,cids,name)