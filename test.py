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


def main():
    df = extract_data('Crime_Data_from_2010_to_Present.csv')
    df = eliminate_unimportant_features(df)
    print(df)


if __name__ == "__main__":
	np.set_printoptions(precision=5)
	main()