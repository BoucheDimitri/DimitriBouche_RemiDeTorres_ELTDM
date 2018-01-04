"""
This is the als_gpu file
which implements the alternating least squares algorithm on gpu
"""


import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as culinalg
culinalg.init()


import kernels
import initialization
import cost_and_errors
import global_variables as gv


def cholesky_solve(a_gpu):
    """
    Performs matrix inversion using Cholesky
    decomposition on gpu using scikit-cuda
    bindings to nvidia cusolver library

    Args :
            a_gpu (pycuda.gpuarray.GPUArray) : the gpuarray version of symmetric matrix to invert

    Returns :
            pycuda.gpuarray.GPUArray. The gpuarray containing the inverse
    """
    acho_gpu = a_gpu.copy()
    culinalg.cholesky(acho_gpu, lib="cusolver")
    acho_inv_gpu = culinalg.inv(acho_gpu, lib="cusolver")
    a_inv_gpu = culinalg.dot(acho_inv_gpu, acho_inv_gpu, transb="T")
    a_inv_gpu = a_inv_gpu.astype(np.float32)
    return a_inv_gpu


def solve_row_subproblem(A_gpu, b_gpu, lamb):
    """
    Solves the least square problem
    min (||A*p  - b||^2 + lamb*||p||^2) in p

    Args :
            A_gpu (pycuda.gpuarray.GPUArray) : A in problem description on gpu
            b_gpu (pycuda.gpuarray.GPUArray) : b in problem description on gpu
            lamb (float) : lamb in problem description

    Returns :
            pycuda.gpuarray.GPUArray. The solution as a gpu array

    """
    A_gpu = A_gpu.astype(np.float32)
    b_gpu = b_gpu.astype(np.float32)
    ncols = A_gpu.shape[1]
    eye_gpu = culinalg.eye(ncols, dtype=np.float32)
    AAT_gpu = culinalg.dot(A_gpu, A_gpu, transa="T")
    AAT_gpu = AAT_gpu.astype(np.float32)
    # print(AAT_gpu.get())
    H_gpu = AAT_gpu + lamb * eye_gpu
    Hinv_gpu = cholesky_solve(H_gpu)
    Hinv_gpu = Hinv_gpu.astype(np.float32)
    # print(Hinv_gpu.get())
    G_gpu = culinalg.dot(Hinv_gpu, A_gpu, transb="T")
    G_gpu = G_gpu.astype(np.float32)
    psol = culinalg.dot(G_gpu, b_gpu)
    return psol.astype(np.float32)


def isolate_usr_subproblem(R_gpu_usr,
                           Q_gpu,
                           usr_no,
                           Nr,
                           Nr_gpu,
                           Nf,
                           Nf_gpu,
                           nrated_usr,
                           nrated_usr_gpu,
                           cum_nrated_usr_gpu,
                           block1=(32, 1, 1),
                           grid1=(2048, 1),
                           block2=(32, 1, 1),
                           grid2=(2048, 1)):
    """
    Get sub matrix of features of movies seen by user
    and set of ratings from user u for those movies

    Args :
            R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by usrs
            Q_gpu (pycuda.gpuarray.GPUArray) : movies features matrix on gpu
            usr_no (int) : the user to consider
            Nr (int) : number of ratings on cpu, redundant but avoids to extract Nr from Nr_gpu to define array shapes
            Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nr on gpu as a (1, 1) array
            Nf (int) : number of features on cpu, redundant but avoids to extract Nf from Nf_gpu to define array shapes
            Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nf on gpu as a (1, 1) array
            nrated_usr (numpy.ndarray) : list of number of ratings for each user as an array
            nrated_usr_gpu (pycuda.gpuarray.GPUArray) : nrated_usr on gpu
            cum_nrated_usr (numpy.ndarray) : np.cumsum(nrated_usr)
            block1 (tuple) : dimension of blocks for execution of kernels.rated_by_usr.function, must be of lenght 3
            grid1 (tuple) : dimension of block grid for execution of kernels.rated_by_usr.function, must be of lenght 2
            block2 (tuple) : dimension of blocks for execution of kernels.subset_sel.function, must be of lenght 3
            grid2 (tuple) : dimension of block grid for execution of kernels.subset_sel.function, must be of lenght 2

    Returns :
            tuple. A tuple of pycuda.gpuarray.GPUArray, (ru, qu) with ru sets of ratings from user u and qu submatrix of features for movies rated by u
    """
    qu = gpuarray.empty((nrated_usr[usr_no], Nf),
                        dtype=np.float32)
    ru = gpuarray.empty((nrated_usr[usr_no], ),
                        dtype=np.float32)
    su = gpuarray.empty((nrated_usr[usr_no], ),
                        dtype=np.uint32)
    indu = gpuarray.to_gpu(np.array([usr_no],
                                    dtype=np.uint32))
    if usr_no != 0:
        indu_minus1 = gpuarray.to_gpu(np.array([usr_no - 1],
                                               dtype=np.uint32))
        beg = gpuarray.take(cum_nrated_usr_gpu, indu_minus1)
    else:
        beg = gpuarray.zeros((1, ), dtype=np.uint32)
    s = gpuarray.take(nrated_usr_gpu, indu)
    kernels.rated_by_usr.function(R_gpu_usr,
                                  ru,
                                  su,
                                  Nr_gpu,
                                  s,
                                  beg,
                                  block=block1,
                                  grid=grid1)
    # print(su.get())
    kernels.subset_sel.function(su,
                                Q_gpu,
                                qu,
                                Nr_gpu,
                                Nf_gpu,
                                s,
                                beg,
                                block=block2,
                                grid=grid2)
    return ru, qu


def isolate_mvi_subproblem(R_gpu_mvi,
                           P_gpu,
                           mvi_no,
                           Nr,
                           Nr_gpu,
                           Nf,
                           Nf_gpu,
                           nratings_mvi,
                           nratings_mvi_gpu,
                           cum_nratings_mvi_gpu,
                           block1=(32, 1, 1),
                           grid1=(2048, 1),
                           block2=(32, 1, 1),
                           grid2=(2048, 1)):
    """
    Get sub matrix of features of movies seen by user
    and set of ratings from user u for those movies

    Args :
            R_gpu_mvi (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by mvis
            P_gpu (pycuda.gpuarray.GPUArray) : usr features matrix on gpu
            mvi_no (int) : the movie to consider
            Nr (int) : number of ratings on cpu, redundant but avoids to extract Nr from Nr_gpu to define array shapes
            Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nr on gpu as a (1, 1) array
            Nf (int) : number of features on cpu, redundant but avoids to extract Nf from Nf_gpu to define array shapes
            Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nf on gpu as a (1, 1) array
            nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
            nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : nratings_mvi on gpu
            cum_nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : cum_nratings_mvi on gpu
            block1 (tuple) : dimension of blocks for execution of kernels.ratings_usr.function, must be of lenght 3
            grid1 (tuple) : dimension of block grid for execution of kernels.ratings_usr.function, must be of lenght 2
            block2 (tuple) : dimension of blocks for execution of kernels.subset_sel.function, must be of lenght 3
            grid2 (tuple) : dimension of block grid for execution of kernels.subset_sel.function, must be of lenght 2

    Returns :
            tuple. A tuple of pycuda.gpuarray.GPUArray, (rm, pm) with rm sets of ratings received by movie m and pm submatrix of features of usrs that have rated m
    """
    pm = gpuarray.empty((nratings_mvi[mvi_no], Nf),
                        dtype=np.float32)
    rm = gpuarray.empty((nratings_mvi[mvi_no], ),
                        dtype=np.float32)
    sm = gpuarray.empty((nratings_mvi[mvi_no], ),
                        dtype=np.uint32)
    indm = gpuarray.to_gpu(np.array([mvi_no],
                                    dtype=np.uint32))
    if mvi_no != 0:
        indm_minus1 = gpuarray.to_gpu(np.array([max(0, mvi_no - 1)],
                                               dtype=np.uint32))
        beg = gpuarray.take(cum_nratings_mvi_gpu, indm_minus1)
    else:
        beg = gpuarray.zeros((1, ), dtype=np.uint32)
    s = gpuarray.take(nratings_mvi_gpu, indm)
    kernels.rating_usrs.function(R_gpu_mvi,
                                 rm,
                                 sm,
                                 Nr_gpu,
                                 s,
                                 beg,
                                 block=block1,
                                 grid=grid1)
    kernels.subset_sel.function(sm,
                                P_gpu,
                                pm,
                                Nr_gpu,
                                Nf_gpu,
                                s,
                                beg,
                                block=block2,
                                grid=grid2)
    return rm, pm


def solve_update_usr(R_gpu_usr,
                     Q_gpu,
                     P_gpu,
                     lamb,
                     usr_no,
                     Nr,
                     Nr_gpu,
                     Nf,
                     Nf_gpu,
                     Nu_gpu,
                     nrated_usr,
                     nrated_usr_gpu,
                     cum_nrated_usr_gpu,
                     block1=(32, 1, 1),
                     grid1=(2048, 1),
                     block2=(32, 1, 1),
                     grid2=(2048, 1),
                     block3=(32, 1, 1),
                     grid3=(1, 1, 1)):
    """
    Solve least squares subproblem for a given usr on gpu, update Q_gpu inplace

    Args :
            R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by usrs
            Q_gpu (pycuda.gpuarray.GPUArray) : mvi features matrix on gpu
            P_gpu (pycuda.gpuarray.GPUArray) : usr features matrix on gpu
            lamb (float) : penalization coefficient
            usrno (int) : user id
            Nr (int) : number of ratings on cpu, redundant but avoids to extract Nr from Nr_gpu to define array shapes
            Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
            Nf (int) : number of features on cpu, redundant but avoids to extract Nf from Nf_gpu to define array shapes
            Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nf on gpu as a (1, 1) array
            Nu_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
            nrated_usr (numpy.ndarray) : list of number of ratings for each user as an array
            nrated_usr_gpu (pycuda.gpuarray.GPUArray) : nrated_usr on gpu
            cum_nrated_usr (numpy.ndarray) : np.cumsum(nrated_usr)
            block1 (tuple) : kernel parameters for solve update functions
            grid1 (tuple) :  Idem as block1
            block2 (tuple) : Idem as block1
            grid2 (tuple) : Idem as block1
            block3 (tuple) : Idem as block1
            grid3 (tuple) : Idem as block1


    Returns :
            Nonetype.
    """
    ru, qu = isolate_usr_subproblem(R_gpu_usr,
                                    Q_gpu,
                                    usr_no,
                                    Nr,
                                    Nr_gpu,
                                    Nf,
                                    Nf_gpu,
                                    nrated_usr,
                                    nrated_usr_gpu,
                                    cum_nrated_usr_gpu,
                                    block1,
                                    grid1,
                                    block2,
                                    grid2)
    psol = solve_row_subproblem(qu, ru, lamb)
    loc = initialization.scalar_to_1d_gpuarray(usr_no,
                                               dtype=np.uint32)
    # print(psol.get())
    kernels.modify_row.function(P_gpu,
                                Nu_gpu,
                                Nf_gpu,
                                psol,
                                loc,
                                block=block3,
                                grid=grid3)

# def solve_update_usr(Q_gpu, P_gpu, )


def solve_update_mvi(R_gpu_mvi,
                     P_gpu,
                     Q_gpu,
                     lamb,
                     mvi_no,
                     Nr,
                     Nr_gpu,
                     Nf,
                     Nf_gpu,
                     Nm_gpu,
                     nratings_mvi,
                     nratings_mvi_gpu,
                     cum_nratings_mvi_gpu,
                     block1=(32, 1, 1),
                     grid1=(2048, 1),
                     block2=(32, 1, 1),
                     grid2=(2048, 1),
                     block3=(32, 1, 1),
                     grid3=(1, 1, 1)):
    """
    Solve least squares subproblem for a given movie on gpu, update P_gpu inplace

    Args :
            R_gpu_mvi (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by mvis
            P_gpu (pycuda.gpuarray.GPUArray) : usr features matrix on gpu
            Q_gpu (pycuda.gpuarray.GPUArray) : mvi features matrix on gpu
            lamb (float) : penalization coefficient
            mvi_no (int) : movie id
            Nr (int) : number of ratings on cpu, redundant but avoids to extract Nr from Nr_gpu to define array shapes
            Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
            Nf (int) : number of features on cpu, redundant but avoids to extract Nf from Nf_gpu to define array shapes
            Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nf on gpu as a (1, 1) array
            Nm_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
            nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
            nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : nratings_mvi on gpu
            cum_nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : cum_nratings_mvi on gpu
            block1 (tuple) : kernel parameters for solve update functions
            grid1 (tuple) :  Idem as block1
            block2 (tuple) : Idem as block1
            grid2 (tuple) : Idem as block1
            block3 (tuple) : Idem as block1
            grid3 (tuple) : Idem as block1

    Returns :
            Nonetype.
    """
    rm, qm = isolate_mvi_subproblem(R_gpu_mvi,
                                    P_gpu,
                                    mvi_no,
                                    Nr,
                                    Nr_gpu,
                                    Nf,
                                    Nf_gpu,
                                    nratings_mvi,
                                    nratings_mvi_gpu,
                                    cum_nratings_mvi_gpu,
                                    block1,
                                    grid1,
                                    block2,
                                    grid2)
    qsol = solve_row_subproblem(qm, rm, lamb)
    loc = initialization.scalar_to_1d_gpuarray(mvi_no,
                                               dtype=np.uint32)
    # print(psol.get())
    kernels.modify_row.function(Q_gpu,
                                Nm_gpu,
                                Nf_gpu,
                                qsol,
                                loc,
                                block=block3,
                                grid=grid3)


def solve_update_all_usrs(R_gpu_usr,
                          Q_gpu,
                          P_gpu,
                          lamb,
                          Nr,
                          Nr_gpu,
                          Nf,
                          Nf_gpu,
                          Nu_gpu,
                          nrated_usr,
                          nrated_usr_gpu,
                          cum_nrated_usr_gpu,
                          block1=(32, 1, 1),
                          grid1=(2048, 1),
                          block2=(32, 1, 1),
                          grid2=(2048, 1),
                          block3=(32, 1, 1),
                          grid3=(1, 1, 1)):
    """
    Solve all usrs least squares subproblems on gpu, update Q_gpu inplace

    Args :
            R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by usrs
            Q_gpu (pycuda.gpuarray.GPUArray) : mvi features matrix on gpu
            P_gpu (pycuda.gpuarray.GPUArray) : usr features matrix on gpu
            lamb (float) : penalization coefficient
            Nr (int) : number of ratings on cpu, redundant but avoids to extract Nr from Nr_gpu to define array shapes
            Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
            Nf (int) : number of features on cpu, redundant but avoids to extract Nf from Nf_gpu to define array shapes
            Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nf on gpu as a (1, 1) array
            Nu_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
            nrated_usr (numpy.ndarray) : list of number of ratings for each user as an array
            nrated_usr_gpu (pycuda.gpuarray.GPUArray) : nrated_usr on gpu
            cum_nrated_usr (numpy.ndarray) : np.cumsum(nrated_usr)
            block1 (tuple) : kernel parameters for solve update functions
            grid1 (tuple) :  Idem as block1
            block2 (tuple) : Idem as block1
            grid2 (tuple) : Idem as block1
            block3 (tuple) : Idem as block1
            grid3 (tuple) : Idem as block1


    Returns :
            Nonetype.
    """
    nusers = nrated_usr.shape[0]
    for u in range(0, nusers):
        solve_update_usr(R_gpu_usr,
                         Q_gpu,
                         P_gpu,
                         lamb,
                         u,
                         Nr,
                         Nr_gpu,
                         Nf,
                         Nf_gpu,
                         Nu_gpu,
                         nrated_usr,
                         nrated_usr_gpu,
                         cum_nrated_usr_gpu,
                         block1,
                         grid1,
                         block2,
                         grid2,
                         block3,
                         grid3)


def solve_update_all_mvis(R_gpu_mvi,
                          P_gpu,
                          Q_gpu,
                          lamb,
                          Nr,
                          Nr_gpu,
                          Nf,
                          Nf_gpu,
                          Nm_gpu,
                          nratings_mvi,
                          nratings_mvi_gpu,
                          cum_nratings_mvi_gpu,
                          block1=(32, 1, 1),
                          grid1=(2048, 1),
                          block2=(32, 1, 1),
                          grid2=(2048, 1),
                          block3=(32, 1, 1),
                          grid3=(1, 1, 1)):
    """
    Solve all mvis least squares subproblems on gpu, update P_gpu inplace

    Args :
            R_gpu_mvi (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by mvis
            P_gpu (pycuda.gpuarray.GPUArray) : usr features matrix on gpu
            Q_gpu (pycuda.gpuarray.GPUArray) : mvi features matrix on gpu
            lamb (float) : penalization coefficient
            Nr (int) : number of ratings on cpu, redundant but avoids to extract Nr from Nr_gpu to define array shapes
            Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
            Nf (int) : number of features on cpu, redundant but avoids to extract Nf from Nf_gpu to define array shapes
            Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nf on gpu as a (1, 1) array
            Nm_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
            nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
            nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : nratings_mvi on gpu
            cum_nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : cum_nratings_mvi on gpu
            block1 (tuple) : kernel parameters for solve update functions
            grid1 (tuple) :  Idem as block1
            block2 (tuple) : Idem as block1
            grid2 (tuple) : Idem as block1
            block3 (tuple) : Idem as block1
            grid3 (tuple) : Idem as block1

    Returns :
            Nonetype.
    """
    nmvis = nratings_mvi.shape[0]
    for m in range(0, nmvis):
        solve_update_mvi(R_gpu_mvi,
                         P_gpu,
                         Q_gpu,
                         lamb,
                         m,
                         Nr,
                         Nr_gpu,
                         Nf,
                         Nf_gpu,
                         Nm_gpu,
                         nratings_mvi,
                         nratings_mvi_gpu,
                         cum_nratings_mvi_gpu,
                         block1,
                         grid1,
                         block2,
                         grid2,
                         block3,
                         grid3)
        # print(m)


def iterate_als(R_gpu_usr,
                R_gpu_mvi,
                R_gpu_test,
                P_gpu,
                Q_gpu,
                lamb,
                Nr,
                Nr_gpu,
                Nrt,
                Nrt_gpu,
                Nf,
                Nf_gpu,
                Nu,
                Nu_gpu,
                Nm,
                Nm_gpu,
                nrated_usr,
                nrated_usr_gpu,
                cum_nrated_usr_gpu,
                nratings_mvi,
                nratings_mvi_gpu,
                cum_nratings_mvi_gpu,
                maxit,
                stopcrit,
                block1=(32, 1, 1),
                grid1=(2048, 1),
                block2=(32, 1, 1),
                grid2=(2048, 1),
                block3=(32, 1, 1),
                grid3=(1, 1, 1),
                block4=(gv.N_THREADSX_SQRNORM, 32, 1),
                grid4=(4096, 1),
                block5=(1024, 1, 1),
                grid5=(4096, 1)):
    """
    Perform several iterations of ALS on GPU. Modify p_gpu and q_gpu inplace

    Args :
            R_gpu_usr (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by usrs
            R_gpu_mvi (pycuda.gpuarray.GPUArray) : rating matrix on gpu sorted by mvis
            R_gpu_test (pycuda.gpuarray.GPUArray) : test data sets, order does not matter
            P_gpu (pycuda.gpuarray.GPUArray) : usr features matrix on gpu
            Q_gpu (pycuda.gpuarray.GPUArray) : mvi features matrix on gpu
            lamb (float) : penalization coefficient
            Nr (int) : number of ratings on cpu, redundant but avoids to extract Nr from Nr_gpu to define array shapes
            Nr_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nr on gpu as a (1, ) array
            Nrt (int) : number of ratings for test dataset on cpu
            Nrt_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nrt on gpu as a (1, 1) array
            Nf (int) : number of features on cpu, redundant but avoids to extract Nf from Nf_gpu to define array shapes
            Nf_gpu (pycuda.gpuarray.GPUArray) : shape=(1, 1), Nf on gpu as a (1, 1) array
            Nu (int) : number of users on cpu
            Nu_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
            Nm (int) : number of movies on cpu
            Nm_gpu (pycuda.gpuarray.GPUArray) : shape=(1, ), Nu on gpu as a (1, ) array
            nrated_usr (numpy.ndarray) : list of number of ratings for each user as an array
            nrated_usr_gpu (pycuda.gpuarray.GPUArray) : nrated_usr on gpu
            cum_nrated_usr (numpy.ndarray) : np.cumsum(nrated_usr)
            nratings_mvi (numpy.ndarray) : list of number of ratings for each movie as an array
            nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : nratings_mvi on gpu
            cum_nratings_mvi_gpu (pycuda.gpuarray.GPUArray) : cum_nratings_mvi on gpu
            maxit (int) : maximum number of iterations
            stop (float) : stopping criterion, will stop when the decrease in the cost is inferior than stop
            block1 (tuple) : kernel parameters for solve update functions
            grid1 (tuple) :  Idem as block1
            block2 (tuple) : Idem as block1
            grid2 (tuple) : Idem as block1
            block3 (tuple) : Idem as block1
            grid3 (tuple) : Idem as block1
            block4 (tuple) : kernel parameters for cost and error kernels
            grid4 (tuple) : Idem as block4
            block5 (tuple) : Idem as block4
            grid5 (tuple) : Idem as block4

    Returns :
            tuple. tuple of size 3 (cost list, mse on learn list, mse on tests list).
    """
    iterno = 0

    costinit, mseinit = cost_and_errors.cost_gpu(R_gpu_usr,
                                                 P_gpu,
                                                 Q_gpu,
                                                 Nr,
                                                 Nr_gpu,
                                                 Nu,
                                                 Nu_gpu,
                                                 Nm,
                                                 Nm_gpu,
                                                 lamb,
                                                 block1=block4,
                                                 grid1=block4,
                                                 block2=block5,
                                                 grid2=block5)

    costs = [float(costinit.get())]
    mses_learn = [float(mseinit.get())]

    mseinittest = cost_and_errors.mse_gpu(R_gpu_test,
                                          P_gpu,
                                          Q_gpu,
                                          Nrt,
                                          Nrt_gpu,
                                          Nu,
                                          Nu_gpu,
                                          block=block5,
                                          grid=grid5)

    mses_test = [float(mseinittest.get())]

    deltacost = costs[0]

    while (iterno < maxit) and (deltacost > stopcrit):
        solve_update_all_usrs(R_gpu_usr,
                              Q_gpu,
                              P_gpu,
                              lamb,
                              Nr,
                              Nr_gpu,
                              Nf,
                              Nf_gpu,
                              Nu_gpu,
                              nrated_usr,
                              nrated_usr_gpu,
                              cum_nrated_usr_gpu,
                              block1,
                              grid1,
                              block2,
                              grid2,
                              block3,
                              grid3)

        solve_update_all_mvis(R_gpu_mvi,
                              P_gpu,
                              Q_gpu,
                              lamb,
                              Nr,
                              Nr_gpu,
                              Nf,
                              Nf_gpu,
                              Nm_gpu,
                              nratings_mvi,
                              nratings_mvi_gpu,
                              cum_nratings_mvi_gpu,
                              block1,
                              grid1,
                              block2,
                              grid2,
                              block3,
                              grid3)

        cost, mse = cost_and_errors.cost_gpu(R_gpu_usr,
                                             P_gpu,
                                             Q_gpu,
                                             Nr,
                                             Nr_gpu,
                                             Nu,
                                             Nu_gpu,
                                             Nm,
                                             Nm_gpu,
                                             lamb,
                                             block1=block4,
                                             grid1=block4,
                                             block2=block5,
                                             grid2=block5)

        costs.append(float(cost.get()))
        mses_learn.append(float(mse.get()))

        msetest = cost_and_errors.mse_gpu(R_gpu_test,
                                          P_gpu,
                                          Q_gpu,
                                          Nrt,
                                          Nrt_gpu,
                                          Nu,
                                          Nu_gpu,
                                          block=block5,
                                          grid=grid5)

        mses_test.append(float(msetest.get()))

        print(iterno)

        deltacost = costs[iterno] - costs[iterno + 1]
        iterno += 1
    return costs, mses_learn, mses_test
