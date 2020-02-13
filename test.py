############ IMPORTS #############
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import sys
##################################


def extract_data(file):
	'''
	Parameters: path to a csv file with dataset 
	
	Return value: data from csv file as pandas object
	'''
	df = pd.read_csv(file)
	df.drop(['Neo Reference ID', 'Name', 'Close Approach Date', 'Epoch Date Close Approach', 'Orbiting Body', 'Orbit ID', 'Orbit Determination Date', 'Equinox'], 1, inplace=True)
	return df


def split_matrix_vector(dataset):
	'''
	Parameters: dataset with outcome column
	
	Return value: X Matrix, y Vector
	'''
	# print(dataset)
	y = np.asarray(dataset['Hazardous'])
	X = np.asarray(dataset.drop('Hazardous',1))
	return X, y


def shuffle_and_split_train_test(X, y, features):
	'''
	Parameters: X-Matrix, y-Vector, Matrix features
	
	Return value: Train dataset and Test dataset
	'''
	obj = { }
	for i in range(len(features)-1):
		print("feature: {}".format(features[i]))
		print("values: {}".format(X[:, i]))
		obj[features[i]] = X[:, i]
	
	obj[len(features)] = y[i]
	df = pd.DataFrame(obj)
	df = shuffle(df)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	return X_train, X_test, y_train, y_test



def rescale_data(data):
	'''
	Rescale data to standard distribution
	'''
	return StandardScaler().fit_transform(data)


def main():
	df = extract_data('./dataset/nasa.csv')
	features = df.columns.values
	print(features)

	# Split to X-Matrix and y-Vector
	X, y = split_matrix_vector(df)
	print("Original dataset shape: {}".format(Counter(y)))
	sm = SMOTE(random_state=42)
	X_res, y_res = sm.fit_resample(X, y)
	print("Resampled dataset shape: {}".format(Counter(y_res)))
	X_train, X_test, y_train, y_test = shuffle_and_split_train_test(X_res, y_res, features)


if __name__ == "__main__":
	np.set_printoptions(precision=5)
	main()