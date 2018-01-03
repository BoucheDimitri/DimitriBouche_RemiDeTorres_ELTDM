"""
This is the data_treatment files which regroups all fuction dealing with pre-treatment of the database of ratings
"""



import pandas as pd 
import numpy as np 
import sys


def load_dataset(path):
	dataset = pd.read_csv(path)
	return dataset


def reindex_from_zero(dataset):
	#Since python indexation goes from 0
	#Better to have indexes starting at 0
	dataset["userId"] -= 1
	dataset["movieId"] -= 1


def get_nusrs(dataset):
	"""
	Get the number of different users from dataset

    Args:
       dataset (pandas.core.frame.DataFrame): The dataset, must contain the col "userId"

    Returns:
       int.  The number of users
	"""
	nusrs = np.unique(dataset["userId"]).shape[0]
	return nusrs


def get_nmvis(dataset):
	"""
	Get the number of different movies from dataset
	
    Args:
       dataset (pandas.core.frame.DataFrame): The dataset, must contain the col "movieId"

    Returns:
       int.  The number of movies
	"""
	nmvis = np.unique(dataset["movieId"]).shape[0]
	return nmvis


def get_nrtgs(dataset):
	"""
	Get the number of ratings in the dataset
	
    Args:
       dataset (pandas.core.frame.DataFrame): The dataset, must contain the col "userId"

    Returns:
       int.  The number of ratings
	"""
	nrtgs = dataset.shape[0]
	return nrtgs


def dataset_to_ints(dataset):
	"""
	When converted to numpy array, it is simpler to have a single dtype
	Since ratings are in {1, 1.5, 2, 2.5 ... 4.5, 5}
	we can get an int dtype for them as well by multiplying by 10
	This is more efficient memorywise since our database can 
	then by a numpy array of ints instead of a numpy array of floats

    Args:
       dataset (pandas.core.frame.DataFrame): The dataset, must contain column "rating"

    Returns:
        pandas.core.frame.DataFrame.  the modified DataFrame
	"""
	dataset["rating"] *= 10
	dataset = dataset.astype(np.int64)
	return dataset


def learn_test_div(dataset, 
				   learnsize=0.7, 
				   resptime=False, 
				   shuffle=True):
	"""
	Divide the dataset into a learnset and a testset

		Args:
			dataset (pandas.core.frame.DataFrame): The dataset of ratings
			learnsize (float): between 0 and 1, the fraction of dataset for learning
			resptime (bool): sort by timestamp before division, time coherence is conserved
			shuffle (bool): shuffle before division, if True time coherence is not conserved

		Returns:
			tuple. A tuple of pandas.core.frame.DataFrame, (learn, test)
	"""
	if resptime:
		processed = dataset.sort_values(by="timestamp", 
										axis=0)
	if shuffle:
		processed = dataset.sample(frac=1)
	nrtgs = get_nrtgs(dataset)
	begtest = int(learnsize*nrtgs)
	test = processed.iloc[begtest:, :]
	learn = processed.iloc[:begtest, :]
	return learn, test


def get_common(learnset, testset):
	"""
	Users and movies must be common between the learnset and testset
	Thus, there is necessarily a loss of users and movies from 
	dividing in learn and test sets. This function returns 
	the common users and common movies set between learnset and testset

	Args:
       learnset (pandas.core.frame.DataFrame): the learn set
       testset (pandas.core.frame.DataFrame): the test set

    Returns:
       tuple. tuple of sets (commonusers, commonmovies)

	"""
	#n of common users for sorted set
	userslearn = set(learnset["userId"])
	userstest = set(testset["userId"])
	#Print n of common movies for sorted set
	movieslearn = set(learnset["movieId"])
	moviestest = set(testset["movieId"])
	#Get the common users and movies
	commonusers = userslearn.intersection(userstest)
	commonmovies = movieslearn.intersection(moviestest)
	return commonusers, commonmovies


def only_common(learnset, testset):
	"""
	Users and movies must be common between the learnset and testset
	Thus, there is necessarily a loss of users and movies from 
	dividing in learn and test sets. This function return 
	modified learnset and testset that have both users 
	and movies strictly in common

	Args:
       learnset (pandas.core.frame.DataFrame): the learn set
       testset (pandas.core.frame.DataFrame): the test set

    Returns:
       tuple. tuple of pandas.core.frame.DataFrame (learnset, testset)

	"""
	commonusers, commonmovies = get_common(learnset, testset)
	commonusers = list(commonusers)
	commonmovies = list(commonmovies)
	commonusers.sort()
	commonmovies.sort()
	learnset = learnset[learnset.movieId.isin(commonmovies)]
	testset = testset[testset.movieId.isin(commonmovies)]
	learnset = learnset[learnset.userId.isin(commonusers)]
	testset = testset[testset.userId.isin(commonusers)]
	return learnset, testset


def get_set_from_col(dataset, col):
	"""
	Get a numpy array of all values found in dataset[col]

	Args:
		dataset (pandas.core.frame.DataFrame): The dataset

	Returns:
		numpy.ndarray. The array of all values found in dataset[col]
	"""
	colset = np.unique(dataset[col])
	colset = np.asarray(colset)
	colset = colset.astype(np.int64)
	return colset


def reindex_usrs_mvies(dataset):
	"""
	Reindex users and movies resp. on a range(0, nusers) and range(0, nmovies)

    Args:
       dataset (pandas.core.frame.DataFrame): The dataset, must contain the columns ["userId", "movieId"]

    Returns:
       pandas.core.frame.DataFrame.  the modified DataFrame
	"""
	#Establish a "bijection" between the set of users
	#and the set of integers from 0 to nusers-1
	users = get_set_from_col(dataset, "userId")
	movies = get_set_from_col(dataset, "movieId")
	#Reindex users and movies according to the above bijection
	dataset["movieBijId"] = dataset["movieId"].apply(
		lambda x: int(np.argwhere(movies==int(x))[0][0]))
	dataset["userBijId"] = dataset["userId"].apply(
		lambda x: int(np.argwhere(users==int(x))[0][0]))
	return dataset


def get_cols_nratings(dataset, col):
	"""
	if col is "userBijId" will return a list with the number 
	of movies each usr has seen, for movies will return a list
	of the number of rating for each movie
	"""
	unique, counts = np.unique(dataset[col], return_counts=True)
	return counts


def dataset_asarray(dataset, dtype=np.int32):
	"""
	Users and movies must be common between dataset[col]the learnset and testset
	Thus, there is necessarily a loss of users and movies from 
	dividing in learn and test sets

	Args:
       dataset (pandas.core.frame.DataFrame): the dataset to convert, must contain cols ["userBijId", "movieBijId", "rating"]

    Returns:
       numpy.ndarray. dataset as a numpy array

	"""
	datamatrix = dataset[["userBijId", 
						  "movieBijId", 
						  "rating"]].as_matrix()
	datamatrix = datamatrix.astype(dtype)
	return datamatrix


