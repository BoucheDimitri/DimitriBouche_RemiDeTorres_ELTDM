"""
This is the initialization file
It regroups two functions for initializations of array on gpu
"""



import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culinalg





def scalar_to_1d_gpuarray(scalar, dtype):
	"""
	initialize a (1, ) array on the gpu which only cell's value is scalar

	Args : 
		scalar (float) : the scalar in question
		dtype (type) : the datatype to use
	"""
	scalar_gpu = np.ones((1, ), dtype=dtype)
	scalar_gpu *= scalar
	scalar_gpu = gpuarray.to_gpu(scalar_gpu)
	return scalar_gpu


def init_features_mat(nrows, ncols, dtype):
	"""
	Initialize a random matrix on gpu according to uniform (0, 1) distribution

	Args : 
		nrows (int) :  the number of rows
		ncols (int) : the number of columns

	Returns : 
		pycuda.gpuarray.GPUArray. The random matrix on gpu
	"""
	mat = np.random.sample((nrows, ncols))
	mat = mat.astype(dtype=dtype)
	mat_gpu = gpuarray.to_gpu(mat)
	return mat_gpu, mat
