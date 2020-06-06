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



def create_name(varlist,taglist):
    """
    creates a file name for a set of parameterw
    e.g.  varlist = [10,2]  tag_list = ['n','m']
    	  output = 'n10-m2'


    Parameters
    ----------
    varlist : list size n 
    taglist : str list size n

    Outputs
    ---------
    run_tag : str
    """
    
    assert len(varlist)==len(taglist)
    run_tag = taglist[0]+str(varlist[0])
    for i in range(1,len(varlist)):
        run_tag+="-"+taglist[i]+str(varlist[i])
    return run_tag



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



def load_skillcraft(train_size=3000):
	"""
	Loads the skillcraft dataset

	Attributes
	1. GameID: Unique ID number for each game (integer)
	2. LeagueIndex: Bronze, Silver, Gold, Platinum, ... coded 1-8 (Ordinal)
	3. Age: Age of each player (integer)
	4. HoursPerWeek: Reported hours spent playing per week (integer)
	5. TotalHours: Reported total hours spent playing (integer)
	6. APM: Action per minute (continuous)
	7. SelectByHotkeys: No. of unit or building selections made using hotkeys per timestamp (cts)
	8. AssignToHotkeys: No. of units or buildings assigned to hotkeys per timestamp (cts)
	9. UniqueHotkeys: No. of unique hotkeys used per timestamp (cts)
	10. MinimapAttacks: No. of attack actions on minimap per timestamp (cts)
	11. MinimapRightClicks: number of right-clicks on minimap per timestamp (cts)
	12. NumberOfPACs: No. of PACs per timestamp (cts)
	13. GapBetweenPACs: Mean duration in milliseconds between PACs (cts)
	14. ActionLatency: Mean latency from the onset of a PACs to their first action in milliseconds (cts)
	15. ActionsInPAC: Mean number of actions within each PAC (cts)
	16. TotalMapExplored: The number of 24x24 game coordinate grids viewed by the player per timestamp (cts)
	17. WorkersMade: No. of SCVs, drones, and probes trained per timestamp (cts)
	18. UniqueUnitsMade: Unique unites made per timestamp (cts)
	19. ComplexUnitsMade: No. of ghosts, infestors, and high templars trained per timestamp (cts)
	20. ComplexAbilitiesUsed: Abilities requiring specific targeting instructions used per timestamp (cts)

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
	path = os.path.join("datasets","skillcraft","SkillCraft1_Dataset.csv")
	data = pd.read_csv(path,sep=',')

	drops = ["GameID"]
	scraft_df = data.drop(drops,axis=1)

	#ids of data with no missing values
	full_data_ids  = (scraft_df!='?').all(axis=1)
	scraft_df  =scraft_df[full_data_ids]

	X = scraft_df.drop("LeagueIndex",axis=1)
	Y = scraft_df.LeagueIndex

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

	


