"""
File: dataload
Description: Contains functions to load and preprocess datasets
Author Lawrence Stewart <lawrence.stewart@ens.fr>
License: Mit License 
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes,load_boston
import os 
np.random.seed(123)


def load_wine(train_size=4000):
	"""
	Loads the white wine dataset

	Parameters
	----------
	train_size : int (train test split)

	Output
	--------
	train : array
	test : array
	train_y : array
	test_y : array
	"""

	#initialisations for dataset
	scaler = MinMaxScaler()  
	path = os.path.join("datasets","wine","winequality-white.csv")
	data = pd.read_csv(path,sep=';')

	#obtain X and Y from dataframe
	X = data.drop(columns='quality')
	Y = data['quality']

	#fit the scaler to X for preprocessing
	scaler.fit(X)

	#split into train and test sets
	train_x,test_x,train_y,test_y = train_test_split(X,Y,
		random_state = 0, stratify = Y,shuffle=True,
		train_size=train_size)

	#scale the data using sklearn Minmax scalar
	train  = scaler.transform(train_x)
	test   = scaler.transform(test_x)

	#reshape y's from (n,) --> (n,1)
	train_y  = pd.DataFrame.to_numpy(train_y).reshape(-1,1).astype(np.float32)
	test_y  = pd.DataFrame.to_numpy(test_y).reshape(-1,1).astype(np.float32)

	return train,test,train_y,test_y


def load_diab(train_size = 375):
	"""
	Loads the diabetes dataset (note this is already prescaled)
	for more information see 
	https://scikit-learn.org/stable/datasets/index.html#boston-dataset

	---------------------------------------------
    -age age in years
    -sex
    -bmi body mass index
    -bp average blood pressure
    -s1 tc, T-Cells (a type of white blood cells)
    -s2 ldl, low-density lipoproteins
    -s3 hdl, high-density lipoproteins
    -s4 tch, thyroid stimulating hormone
    -s5 ltg, lamotrigine
    -s6 glu, blood sugar level
	---------------------------------------------

	Parameters
	----------
	train_size : int

	Output
	--------
	train : array
	test : array
	train_y : array
	test_y : array
	"""

	X,Y = load_diabetes(return_X_y=True)
	Y = Y.reshape(-1,1)

	train,test,train_y,test_y  = train_test_split(X,Y,random_state=0,
		shuffle=True,train_size = train_size)
	return train,test,train_y,test_y


def load_housing(train_size = 450):
	"""
	Loads boston dataset and applies scaling, splitting into 
	train and test data.

	---------------------------------------------
    -CRIM per capita crime rate by town
    -ZN proportion of residential land zoned for lots over 25,000 sq.ft.
    -INDUS proportion of non-retail business acres per town
    -CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    -NOX nitric oxides concentration (parts per 10 million)
    -RM average number of rooms per dwelling
    -AGE proportion of owner-occupied units built prior to 1940
    -DIS weighted distances to five Boston employment centres
    -RAD index of accessibility to radial highways
    -TAX full-value property-tax rate per $10,000
    -PTRATIO pupil-teacher ratio by town
    -B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    -LSTAT % lower status of the population
    -MEDV Median value of owner-occupied homes in $1000’s (target var)
	---------------------------------------------

	Parameters
	----------
	train_size : int

	Output
	--------
	train : array
	test : array
	train_y : array
	test_y : array
	"""

	X,Y = load_boston(return_X_y=True)
	Y = Y.reshape(-1,1)

	#fit scaler to dataset
	scaler = MinMaxScaler() 
	scaler.fit(X)

	#split into train and test sets
	train_x,test_x,train_y,test_y = train_test_split(X,Y,
		random_state = 0,shuffle=True,
		train_size=train_size)

	#scale the data using sklearn Minmax scalar
	train  = scaler.transform(train_x)
	test   = scaler.transform(test_x)

	return train,test,train_y,test_y




