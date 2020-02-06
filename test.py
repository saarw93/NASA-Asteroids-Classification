############ IMPORTS #############
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import sys
##################################


def extract_data(file):
	'''
	Parameters: xlsx file 
	
	Return value: data from xlsx file as pandas object
	'''
	df = pd.read_csv(file)
	return df


def eliminate_unimportant_features(dataset):
	'''
	Parameters: dataset with X columns
	
	Return value: dataset with Y columns after removing unimportant features (like constants etc.) 
	'''


def split_train_test(df, outcomes, train_percent=0.7):
	'''
	Parameters: dataset, outcome classes, train percent of dataset
	
	Return value: train set, test set
	'''
	train_size = []
	df_outcomes = []
	for outcome in outcomes:
		df_outcomes.append(df.loc[df['AREA'] == outcome])
		train_size.append(int(len(df.loc[df['AREA'] == outcome]) * train_percent))
	
	df_train = pd.DataFrame()
	df_test = pd.DataFrame()
	
	for i in range(len(df_outcomes)):
		df_train = df_train.append(df_outcomes[i][:train_size[i]])
		df_test = df_test.append(df_outcomes[i][train_size[i]:])
	
	
	X_train = np.asarray(df_train.drop('AREA', 1))
	y_train = np.asarray(df_train['AREA'])
	
	X_test = np.asarray(df_test.drop('AREA', 1))
	y_test = np.asarray(df_test['AREA'])
	
	# X_train = rescale_data(X_train)
	# X_test = rescale_data(X_test)
	
	return X_train, X_test, y_train, y_test
	''' End function '''


def rescale_data(data):
	'''
	Rescale data to standard distribution
	'''
	return StandardScaler().fit_transform(data)


def main():
	df = extract_data('./dataset/nasa.csv')
	df = shuffle(df)
	print(df)


if __name__ == "__main__":
	np.set_printoptions(precision=5)
	main()