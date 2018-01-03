"""
This is the cost_and_errors files
It regroups the implementation of errors and cost related functions on the gpu
"""



import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import kernels
import skcuda.linalg as culinalg
import skcuda.misc as misc
import cpu_version
import global_variables as gv

culinalg.init()







def row_norms(A_gpu, 
			  Nrows, 
			  Nrows_gpu, 
			  block=(gv.N_THREADSX_SQRNORM, 32, 1), 
			  grid=(4096, 1)):
	"""
	Compute the vector which cells are the sqr norms of the rows of A_gpu on the gpu

	Args :
		A_gpu (pycuda.gpuarray.GPUArray) : input matrix on gpu
		Nrows (int) : number of rows of A on cpu
		Nrows_gpu (pycuda.gpuarray.GPUArray) : number of rows of A on gpu as a (1, ) gpuarray, 
											   redundant but avoids comings and goings between cpu and gpus 
											   which may prove costly since this function is used in several loops
		block (tuple) : dimension of blocks of threads for kernels.sqr_norm kernel
		grid (tuple) : dimension of grid of threads for kernels.sqr_norm kernel
	"""

	Anorm_gpu = gpuarray.zeros((Nrows, ), dtype=np.float32)
	kernels.sqr_norm.function(A_gpu, 
							  Anorm_gpu, 
							  Nrows_gpu, 
							  block=block, 
							  grid=grid)
	return Anorm_gpu


def sqr_err_vect(R_gpu, 
				 p_gpu, 
				 q_gpu, 
				 Nr, 
				 Nr_gpu, 
				 Nu,
				 Nu_gpu, 
				 block=(1024, 1, 1), 
				 grid=(4096, 1)):
	"""
	Compute vector of square errors on GPU

	Args :
		R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu (order is not important)
		p_gpu (pycuda.gpuarray.GPUArray) : users features matrix on gpu
		q_gpu (pycuda.gpuarray.GPUArray) : movies features matrix on gpu
		Nr (int) : number of ratings on cpu
		Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
		Nf (int) : number of features on cpu
		Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nf on gpu as a (1, ) array
		nrated_usr (numpy.ndarray) : list of number of ratings for each user as an array
		block1 (tuple) : dimension of blocks for execution of kernels.sqr_err_vect.function, must be of lenght 3
		grid1 (tuple) : dimension of block grid for executation of kernels.sqr_err_vect.function, must be of lenght 2
	
	Returns :
		pycuda.gpuarray.GPUArray. Vector of square errors on GPU
	"""

	qpt_gpu = culinalg.dot(q_gpu, p_gpu, transb="T")
	errv = gpuarray.zeros((Nr, ), dtype=np.float32)
	kernels.sqr_err_vect.function(R_gpu, 
								  qpt_gpu, 
								  Nr_gpu, 
								  Nu_gpu, 
								  errv, 
								  block=block, 
								  grid=grid)
	return errv


def pen_vect(R_gpu, 
			 p_gpu, 
			 q_gpu, 
			 Nr,
			 Nr_gpu, 
			 Nu,
			 Nu_gpu, 
			 Nm,
			 Nm_gpu, 
			 block=(gv.N_THREADSX_SQRNORM, 32, 1),
			 grid=(4096, 1)):
	"""
	Compute vector of norm penalities for the cost function

	Args :
		R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu (order is not important)
		p_gpu (pycuda.gpuarray.GPUArray) : users features matrix on gpu
		q_gpu (pycuda.gpuarray.GPUArray) : movies features matrix on gpu
		Nr (int) : number of ratings on cpu
		Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
		Nu (int) : number of users on cpu
		Nu_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
		Nm (int) : number of movies on cpu
		Nm_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
		block (tuple) : dimension of blocks for execution of kernels.pen_vect.function, must be of lenght 3
		grid (tuple) : dimension of block grid for executation of kernels.pen_vect.function, must be of lenght 2
	
	Returns :
		pycuda.gpuarray.GPUArray. Vector of penalization
	"""
	qnorms_gpu = row_norms(q_gpu, Nm, Nm_gpu, block, grid)
	pnorms_gpu = row_norms(p_gpu, Nu, Nu_gpu, block, grid)
	penv = gpuarray.zeros((Nr, ), dtype=np.float32)
	kernels.pen_vect.function(R_gpu, 
							  pnorms_gpu, 
							  qnorms_gpu, 
							  Nr_gpu, penv, 
							  block=block, 
							  grid=grid)
	return penv


def mse_gpu(R_gpu, 
			p_gpu, 
			q_gpu, 
			Nr, 
			Nr_gpu, 
			Nu,
			Nu_gpu, 
			block=(1024, 1, 1), 
			grid=(4096, 1)):
	"""
	Compute mean square error on GPU

	Args :
		R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu (order is not important)
		p_gpu (pycuda.gpuarray.GPUArray) : users features matrix on gpu
		q_gpu (pycuda.gpuarray.GPUArray) : movies features matrix on gpu
		Nr (int) : number of ratings on cpu
		Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
		Nu (int) : number of users on cpu
		Nu_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
		block (tuple) : dimension of blocks for execution of kernels.pen_vect.function, must be of lenght 3
		grid (tuple) : dimension of block grid for executation of kernels.pen_vect.function, must be of lenght 2
	
	Returns :
		pycuda.gpuarray.GPUArray. the mean square error on gpu as a (1, ) gpu array
	"""
	errv = sqr_err_vect(R_gpu, 
						p_gpu, 
						q_gpu, 
						Nr, 
						Nr_gpu, 
						Nu, 
						Nu_gpu, 
						block=block, 
						grid=grid)
	return misc.mean(errv)
	


def cost_gpu(R_gpu, 
			 p_gpu, 
			 q_gpu, 
			 Nr,
			 Nr_gpu, 
			 Nu,
			 Nu_gpu, 
			 Nm,
			 Nm_gpu, 
			 lamb, 
			 block1=(gv.N_THREADSX_SQRNORM, 32, 1),
			 grid1=(4096, 1),
			 block2=(1024, 1, 1),
			 grid2=(4096, 1)):
	"""
	Compute vector of norm penalities for the cost function

	Args :
		R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu (order is not important)
		p_gpu (pycuda.gpuarray.GPUArray) : users features matrix on gpu
		q_gpu (pycuda.gpuarray.GPUArray) : movies features matrix on gpu
		Nr (int) : number of ratings on cpu
		Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
		Nu (int) : number of users on cpu
		Nu_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
		Nm (int) : number of movies on cpu
		Nm_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
		lamb (float) : penalization coefficient
		block1 (tuple) : dimension of blocks for execution of function pen_vect
		grid1 (tuple) : dimension of block grid for execution of function pen_vect
		block2 (tuple) : dimension of blocks for execution of function sqr_err_vect
		grid2 (tuple) : dimension of block grid for execution of function sqr_err_vect

	Returns :
		tuple. (cost a a (1, ) gpuarray, mean square error as a (1, ) gpu array)
	"""
	errv = sqr_err_vect(R_gpu, 
						p_gpu, 
						q_gpu, 
						Nr, 
						Nr_gpu, 
						Nu, 
						Nu_gpu, 
						block=block2, 
						grid=grid2)
	penv = pen_vect(R_gpu, 
			 		p_gpu, 
			 		q_gpu, 
			 		Nr,
			 		Nr_gpu, 
					Nu,
					Nu_gpu, 
					Nm,
					Nm_gpu, 
					block=block1,
					grid=grid1)
	costvec = errv + lamb * penv
	return gpuarray.sum(costvec), misc.mean(errv)


