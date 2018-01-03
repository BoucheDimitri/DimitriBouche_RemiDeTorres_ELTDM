"""
This is the cpu_version file which implements an als based recommender system on the cpu
"""





import numpy as np 


def cost_and_mse(dataset, 
		 	     p, 
		 	     q, 
		 	     lamb):
	"""
	Compute cost and rmse at the same time
	avoids to run the long loop twice 
	when we need both and does not 
	add complexity in comparison with
	computing only one

    Args:
       dataset (numpy.ndarray): The dataset, 
       							dataset.shape = (nratings, 3)
       							dataset[:, 0] = usrIds
       							dataset[:, 1] = mviIds
       							dataset[:, 2] = ratings
       p (numpy.ndarray): the users' features
       					  p.shape = (nfeatures, nusers)
       q (numpy.ndarray): the movies' features
       					  p.shape = (nfeatures, nmovies)
       lamb (float): the regularization coefficient

    Returns:
       tuple.  tuple of float (cost, rmse)
	"""
	cost = 0.0
	mse = 0.0
	nratings = dataset.shape[0]
	for i in range(0, nratings):
		usr = int(dataset[i, 0])
		mvi = int(dataset[i, 1])
		r = 0.1*dataset[i, 2]
		hatr = np.dot(q[mvi, :], 
					  np.transpose(p[usr, :]))
		reg = lamb*(np.power(np.linalg.norm(p[usr, :]), 2) 
					+ np.power(np.linalg.norm(q[mvi, :]), 2))
		err = np.power(r - hatr, 2) 
		cost += err + reg
		mse += err
	mse /= nratings
	return cost, mse


def isolate_usr_subproblem(dataset_usr_order, 
			   		 	   q,
			   		       nrated_usr,
			   		       usr_no):
	"""
	Isolate sub matrix of features and sub vector of ratings
	for usr sub least square problem.

	Args : 
		dataset_usr_order (numpy.ndarray): matrix array ordered by user
		q (numpy.ndarray) :  matrix of movies features
		nrated_usr (numpy.ndarray) : list of number of ratings for each user as an 
		usr_no (int) : id of the user to consider

	Returns : 
		tuple. (subvector of ratings, submatrix of features)
	"""
	cum_nrated_usr = np.cumsum(nrated_usr)
	nrated = nrated_usr[usr_no]
	nf = q.shape[1]
	if usr_no != 0:
		beg = cum_nrated_usr[usr_no - 1]
	else :
		beg = 0
	ru = dataset_usr_order[beg: beg + nrated, 2]
	ru = ru.reshape((nrated, 1))
	mvis = dataset_usr_order[beg: beg + nrated, 1]
	qu = np.empty((nrated, nf), dtype=np.float32)
	for i in range(0, nrated):
		qu[i, :] = q[mvis[i], :]
	return 0.1*ru, qu


def isolate_mvi_subproblem(dataset_mvi_order, 
			   		 	   p,
			   		       nratings_mvi,
			   		       mvi_no):
	"""
	Isolate sub matrix of features and sub vector of ratings
	for mvi sub least square problem. 

	Args : 
		dataset_mvi_order (numpy.ndarray): matrix array ordered by movie
		p (numpy.ndarray) :  matrix of users features
		nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
		mvi_no (int) : id of the movie to consider

	Returns : 
		tuple. (subvector of ratings, submatrix of features)
	"""
	cum_nratings_mvi = np.cumsum(nratings_mvi)
	nratings = nratings_mvi[mvi_no]
	nf = p.shape[1]
	if mvi_no != 0:
		beg = cum_nratings_mvi[mvi_no - 1]
	else :
		beg = 0
	rm = dataset_mvi_order[beg: beg + nratings, 2]
	rm = rm.reshape((nratings, 1))
	usrs = dataset_mvi_order[beg: beg + nratings, 0]
	pm = np.empty((nratings, nf), dtype=np.float32)
	for i in range(0, nratings):
		pm[i, :] = p[usrs[i], :]
	return 0.1*rm, pm
	


def solve_update_usr(dataset_usr_order, 
			   		 q,
			   		 p,
			   		 nrated_usr,
			   		 usr_no, 
			   		 lamb) :
	"""
	Isolate sub matrix of features and sub vector of ratings
	for usr and solve least square associated problem. modify p inplace consequently

	Args : 
		dataset_usr_order (numpy.ndarray): matrix array ordered by user
		q (numpy.ndarray) :  matrix of movies features
		p (numpy.ndarray) :  matrix of users features
		nrated_usr (numpy.ndarray) : list of number of ratings for each user as an 
		usr_no (int) : id of the user to consider
		lamb (float) : penalization coefficient

	Returns : 
		Nonetype. None
	"""
	nf = p.shape[1]
	ru, qu = isolate_usr_subproblem(dataset_usr_order, 
									q, 
									nrated_usr,
									usr_no)
	grammat = np.dot(qu.T, qu)
	toinv = grammat + lamb*np.eye(nf)
	inv = np.linalg.inv(toinv)
	psol = np.dot(np.dot(inv, qu.T), ru)
	p[usr_no, :] = psol.reshape((nf, ))



def solve_update_mvi(dataset_mvi_order, 
			   		 p,
			   		 q,
			   		 nratings_mvi,
			   		 mvi_no, 
			   		 lamb) :
	"""
	Isolate sub matrix of features and sub vector of ratings
	for mvi and solves associated least squares problem, modify q inplace consequently

	Args : 
		dataset_mvi_order (numpy.ndarray): matrix array ordered by movie
		p (numpy.ndarray) :  matrix of users features
		nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
		mvi_no (int) : id of the movie to consider
		lamb (float) : penalization coefficient

	Returns : 
		Nonetype. None
	"""
	nf = p.shape[1]
	rm, pm = isolate_mvi_subproblem(dataset_mvi_order, 
									p, 
									nratings_mvi,
			   		       			mvi_no)
	grammat = np.dot(pm.T, pm)
	toinv = grammat + lamb*np.eye(nf)
	inv = np.linalg.inv(toinv)
	qsol = np.dot(np.dot(inv, pm.T), rm)
	q[mvi_no, :] = qsol.reshape((nf, ))


def solve_update_all_usrs(dataset_usr_order, 
			   		 	  q,
			   		 	  p,
			   		 	  nrated_usr,			   		 	
			   		 	  lamb):
	"""
	Iteration over usr of solve_update_usr, modify p inplace consequently

	Args : 
		dataset_usr_order (numpy.ndarray): matrix array ordered by user
		q (numpy.ndarray) :  matrix of movies features
		p (numpy.ndarray) :  matrix of users features
		nrated_usr (numpy.ndarray) : list of number of ratings for each user as an 
		lamb (float) : penalization coefficient

	Returns : 
		Nonetype. None
	"""
	for u in range(0, nrated_usr.shape[0]):
		solve_update_usr(dataset_usr_order, 
			   		 	 q,
			   		 	 p,
			   		 	 nrated_usr,
			   		 	 u, 
			   		 	 lamb)


def solve_update_all_mvis(dataset_mvi_order, 
			   		 	  p,
			   		 	  q,
			   		 	  nratings_mvi,
			   		 	  lamb):
	"""
	Iteration over mvi of solve_update_mvi, modify q inplace consequently

	Args : 
		dataset_mvi_order (numpy.ndarray): matrix array ordered by movie
		p (numpy.ndarray) :  matrix of users features
		nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
		mvi_no (int) : id of the movie to consider
		lamb (float) : penalization coefficient

	Returns : 
		Nonetype. None
	"""
	for m in range(0, nratings_mvi.shape[0]):
		solve_update_mvi(dataset_mvi_order, 
			   		 	 p,
			   		 	 q,
			   		 	 nratings_mvi,
			   		 	 m, 
			   		 	 lamb)


def iter_solve_update(dataset_usr_order, 
					  dataset_mvi_order,
					  dataset_test,
					  p, 
					  q, 
					  nrated_usr, 
					  nratings_mvi,
					  lamb,
					  maxit,
					  stop):
	"""
	Several iterations of the succession of solve_update_all_usrs and solve_update_all_mvis

	Args : 
		dataset_usr_order (numpy.ndarray): matrix array ordered by user
		dataset_mvi_order (numpy.ndarray): matrix array ordered by movie
		p (numpy.ndarray) :  matrix of users features
		q (numpy.ndarray) :  matrix of movies features
		nrated_usr (numpy.ndarray) : list of number of ratings for each user as an array
		nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
		lamb (float) : penalization coefficient
		maxit (int) : max number of iterations
		stop (float) : stopping criterion, will stop when the decrease in cost < stop

	Returns : 
		Nonetype. None
	"""
	
	itno = 0
	costinit, mseinit = cost_and_mse(dataset_usr_order, p, q, lamb)
	costs = [costinit]
	mses = [mseinit]
	deltacost = costinit
	costinit_test, mseinit_test = cost_and_mse(dataset_test, p, q, lamb)
	mses_test = [mseinit_test]

	while (itno < maxit) and (deltacost > stop):
		solve_update_all_usrs(dataset_usr_order, q, p, nrated_usr, lamb)
		solve_update_all_mvis(dataset_mvi_order, p, q, nratings_mvi, lamb)
		cost, mse = cost_and_mse(dataset_usr_order, p, q, lamb)
		costs.append(cost)
		mses.append(mse)
		cost_test, mse_test = cost_and_mse(dataset_test, p, q, lamb)
		mses_test.append(mse_test)
		deltacost = costs[itno] - costs[itno + 1]
		print(itno)
		itno += 1
	return costs, mses, mses_test


