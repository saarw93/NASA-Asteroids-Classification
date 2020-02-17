############ IMPORTS #############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import sys
import scipy.stats as stats
import math

from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, scale

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.utils import shuffle

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
##################################


class Model:
	'''
	This class holds all the data for a specific Model
	'''
	def __init__(self, modelName, model):
		self.modelName = modelName
		self.model = model
		self.overall = 0

	def calc_overall(self, overall, value):
		self.overall = overall + value



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
	y = np.asarray(dataset['Hazardous'])
	X = np.asarray(dataset.drop('Hazardous',1))
	X = rescale_data(X)
	return X, y


def shuffle_and_split_train_test(X, y, features):
	'''
	Parameters: X-Matrix, y-Vector, Matrix features
	
	Return value: Train dataset and Test dataset
	'''
	obj = { }
	for i in range(len(features)-1):
		# print("feature: {}".format(features[i]))
		# print("values: {}".format(X[:, i]))
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
	return scale(data)
	# return StandardScaler().fit_transform(data)


def get_k_selected_features_names(indices, features):
	'''
	Parameters: features indices in csv, features names
	
	Return value: list of the k selected features names
	'''	
	selected_features = []
	for index in indices:
		selected_features.append(features[index])
	return selected_features


def compare_models(X, y, models):
	'''
	Input: X-Matrix, y-Vector, list of models to compare

	Return value: The two best models with the best measurements 
	'''
	measurements = ['accuracy', 'precision', 'recall', 'roc_auc']
	for i in range(len(measurements)):
		print('--------------------------------')
		print('Compare algorithms for', measurements[i])
		results = []
		means = []
		stds = []
		for model in models:
			cv_result = cross_val_score(model.model, X, y, cv=10, scoring=measurements[i], n_jobs=-1)
			results.append(cv_result)
			mean = cv_result.mean()
			std = cv_result.std()
			means.append(mean)
			stds.append(std)
			print("Algorithm: {}. Mean: {}. Std:{}".format(model.modelName, cv_result.mean(), cv_result.std()))
			if i == 0:
				model.calc_overall(model.overall, mean * 0.15)
			elif i == 0:
				model.calc_overall(model.overall, mean * 0.15)
			elif i == 2:
				model.calc_overall(model.overall, mean * 0.25)
			elif i == 3:
				model.calc_overall(model.overall, mean * 0.45)
		colors = ['r', 'g', 'b', 'y', 'c']
		for j in range(len(results)):
			mu = results[j].mean()
			variance = results[j].std()
			sigma = math.sqrt(variance)
			x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
			plt.plot(x, stats.norm.pdf(x, mu, sigma), color=colors[j], label=models[j].modelName)
			plt.title("Algorithm {} Comparison".format(measurements[i]))
			plt.xlabel("Normal Distribution")
			plt.legend(loc="upper left")
		plt.show()

	models.sort(key=lambda model: model.overall, reverse=True)
	if models[0].modelName is 'DT':
		return models[1:3]
	return model[0:2]


################################################################################
##################################### MAIN #####################################
################################################################################
def main():
	df = extract_data('./dataset/nasa.csv')
	df = shuffle(df)
	features = df.columns.values
	print(features)


	# Split to X-Matrix and y-Vector
	X, y = split_matrix_vector(df)
	print("Original dataset shape: {}".format(Counter(y)))
	print("Number of samples: {}".format(X.shape[0]))
	print("Number of features: {}".format(X.shape[1]))
	print("Ratio between classes: {}".format(y[y == True].shape[0] / y[y == False].shape[0]))
	

	# Upscale data with SMOTE algorithm - ratio between classes is 1:2
	# For example: over 2 samples of non-hazardous asteroids there is 1 sample of hazardous asteroid 
	sm = SMOTE(sampling_strategy=0.5, random_state=42)
	X_res, y_res = sm.fit_resample(X, y)
	print("Resampled dataset shape after SMOTE: {}".format(Counter(y_res)))
	print("Number of samples: {}".format(X_res.shape[0]))
	print("Number of features: {}".format(X_res.shape[1]))
	print("Ratio between classes: {}".format(y_res[y_res == True].shape[0] / y_res[y_res == False].shape[0]))
	
	# Downscale data with NearMiss algorithm - ratio between classes is 1:1
	nm = NearMiss()
	X_res, y_res = nm.fit_resample(X_res, y_res)
	print("Resampled dataset shape after NearMiss: {}".format(Counter(y_res)))
	print("Number of samples: {}".format(X_res.shape[0]))
	print("Number of features: {}".format(X_res.shape[1]))
	print("Ratio between classes: {}".format(y_res[y_res == True].shape[0] / y_res[y_res == False].shape[0]))	


	# Select the best 15 features that gives the best indication of y
	selector = SelectKBest(f_classif, k=15)
	X_res = selector.fit_transform(X_res, y_res)
	
	# Get columns to indentify which features were seleted by SelectKBest
	selected_features_indices = selector.get_support(indices=True)
	print(selected_features_indices)
	selected_features_names = get_k_selected_features_names(selected_features_indices, features)
	print("The selected features are: {}".format(selected_features_names))

	print("Dataset shape after feature selection: {}".format(Counter(y_res)))
	print("Number of samples: {}".format(X_res.shape[0]))
	print("Number of features: {}".format(X_res.shape[1]))
	print("Ratio between classes: {}".format(y_res[y_res == True].shape[0] / y_res[y_res == False].shape[0]))
	

	# Prepare models:
	models = []
	models.append(Model("LR", LogisticRegression(solver='liblinear', max_iter=10**6)))
	models.append(Model("SVC", SVC(gamma='auto')))
	models.append(Model("KNN", KNeighborsClassifier()))
	models.append(Model("GNN", GaussianNB()))
	models.append(Model("DT", DecisionTreeClassifier()))

	best_models = compare_models(X_res, y_res, models)
	for model in best_models:
		print(model.modelName)


if __name__ == "__main__":
	np.set_printoptions(precision=5)
	main()