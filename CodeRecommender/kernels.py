"""
This is the kernels file which regroups cuda c scripts of kernels used in the project
Those scripts are compiled as python function at the end of the file using pycuda
"""


import global_variables as gv
import Kernel


# Dictionnary storing the different kernel code by
# their signature in the C code
kernel_dic = {}


#######################KERNEL#SCRIPTS#####################################
##########################################################################

kernel_dic["SqrNormRowsKernel"] = """
__global__ void SqrNormRowsKernel(float *Q,
                                  float *Qnorm,
                                  uint *Nrows)

//Compute the vector of square norms of rows
//of a matrix using 2D blocks

//Designed for matrix that have a small
//number of columns in comparison with
//their number of rows :

//IMPORTANT : WILL NOT WORK IF Ncols > blockDim.y

//Thus works for our features matrix since
//the number of features is typically
//small enough to have blockDim.y >= N_FEATURES

{
    const uint Ncols = %(N_COLUMNS)s;
    const uint Nthreadsx = %(N_THREADS_X)s;

    //const uint cacheSize = Nthreadsx * Ncols;

    uint bx = blockIdx.x;
    uint tx = threadIdx.x;
    uint ty = threadIdx.y;

    __shared__ float cache[Nthreadsx][Ncols];

    //Since each block does the computations
    //for blockDim.x rows, we are finished
    // when bx * blockDim.x < Nrows
    while (bx * blockDim.x < Nrows[0]) {

        //ty is used to browse through the columns of the matrix
        if (ty < Ncols){

            float q = Q[(bx * Nthreadsx + tx) * Ncols + ty];
            
            cache[tx][ty] = q * q;
        }

        //Wait for each thread to be finished
        //to sum the results
        __syncthreads();

        //First thread of each line of the block does the summing
        if (ty == 0){

             float sum = 0;

             for (int i = 0; i < Ncols; i++)
                 sum += cache[tx][i];

             //write the sum results in right spot of Qnorm
             Qnorm[bx * blockDim.x + tx] = sum;

        }

        //In case gridDim.x * blockDim.x < Ncols
        bx += gridDim.x;
    }
}
"""


kernel_dic["ModifyRowKernel"] = """
__global__ void ModifyRowKernel(float *Q,
                                uint *Nrows,
                                uint *Ncols,
                                float *q,
                                int *loc)

//Kernel to modify a given row of a matrix inplace
//Q is the matrix to modify
//Nrows : Q's number of rows
//Ncols : Q's number of columns
//loc : index of the row to modify
//q : data to replace row with

{    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < Ncols[0]){
    
        Q[loc[0] * Ncols[0] + tid] = q[tid];
        //In case we have Nthreadsx < Ncols
        tid += blockDim.x * gridDim.x;
    }
}
"""


kernel_dic["SqrErrVectKernel"] = """
__global__ void SqrErrVectKernel(int *R,
                                 float *QPT,
                                 uint *N,
                                 uint *Nu,
                                 float *sqrerr)

//Compute the square error of predicted ratings compared to true ones
//R is the matrix of ratings
//QPT is the scalar product between Q and PT
//N is the number of ratings
//Nu is number of users
//sqerr is the container to get the result

{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N[0]) {

        int usr = R[tid];
        int mvi = R[N[0] + tid];
        int rtg = R[2*N[0] + tid];

        //Recreate the original grade range from 1 to 5
        float rtgf;
        rtgf = (float) 0.1*rtg;

        //Compute error and store in sqerr
        //float err = rtgf - QPT[usr * Nm[0] + mvi];
        float err = rtgf - QPT[mvi * Nu[0] + usr];
        sqrerr[tid] = err*err;

        //In case blockDim.x*gridDim.x < N[0]
        tid += blockDim.x * gridDim.x;
        
    }

}
 """


kernel_dic["PenVectKernel"] = """
__global__ void PenVectKernel(int *R,
                              float *Pnorm,
                              float *Qnorm,
                              uint *N,
                              float *pen)

// Compute penalization vector
// R is the matrix of ratings
// Pnorm is the vector of norm of rows of P
// Qnorm is the vector of norm of rows of Q
// N is the number of ratings
// pen is the container to write the result

{
    uint tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N[0]) {

        int usr = R[tid];
        int mvi = R[N[0] + tid];
        pen [tid] = Pnorm[usr] + Qnorm[mvi];

        //In case blockDim.x*gridDim.x < N[0]
        tid += blockDim.x * gridDim.x;
    }
}
 """


kernel_dic["RatedByUsrKernel"] = """
__global__ void RatedByUsrKernel(uint *R,
                                 float *Ru,
                                 uint *Su,
                                 uint *N,
                                 uint *s,
                                 uint *beg)

//Select submatrix of R for the set of movie rated by a given user
//Ru is the container for ratings subvector
//Su is the container for vector of movies rated by u
//N is the total number of ratings
//s is the number of movies rated by user u
//beg is the index of first rating from user u

{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < s[0]){
    
        Su[tid] = R[N[0] + beg[0] + tid];
        Ru[tid] = 0.1 * R[2 * N[0] + beg[0] + tid];
        tid += blockDim.x * gridDim.x;
    }
}
 """


kernel_dic["MviRatingUsrsKernel"] = """
__global__ void MviRatingUsrsKernel(uint *R,
                                    float *Rm,
                                    uint *Sm, 
                                    uint *N,
                                    uint *r,
                                    uint *beg)

//Select submatrix of R for the set of usrs who rated a given movie
//Rm is the container for ratings subvector
//Sm is the container for vector of users rating the movie
//N is the total number of ratings
//s is the number of ratings for the movie
//beg is the index of first user to rate the movie

{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < r[0]){
    
        Sm[tid] = R[beg[0] + tid];
        Rm[tid] = 0.1 * R[2 * N[0] + beg[0] + tid];
        tid += blockDim.x * gridDim.x;
    }
}
 """


kernel_dic["SubsetSelKernel"] = """
__global__ void SubsetSelKernel(uint *Ss,
                                float *Q,
                                float *Qs,
                                uint *N,
                                uint *Nf,
                                uint *s,
                                uint *beg)

//Select a sub matrix of features for a given set of movies or of users
//Ss is either a set of movies in usr case either a set of usrs in mvi case
//Q is either Q in usr case or P in movie case
//Qs is the container to receive the submatrix of features
//s is the number of seen movies by user u in user case
//or the number of ratings received by movie m in movie case
//beg is the index of first rating from user u in user case
//beg is the index of first user to rate m in movie case

{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while(tid < s[0] * Nf[0]){
        uint mcount = tid / Nf[0];
        
        //modulo operator is not supported thus we do
        //the following to get the remains in integer division
        
        uint fcount = tid - mcount * Nf[0];
        Qs[tid] = Q[Ss[mcount] * Nf[0] + fcount];

        tid += blockDim.x * gridDim.x;
    }
}
 """


####################COMPILATION#OF#SCRIPTS####################################
##############################################################################


# Modify row of matrix inplace kernel
modify_row = Kernel.Kernel("ModifyRowKernel",
                           kernel_dic["ModifyRowKernel"])
modify_row.compile()


# array of sqr norm of rows of a matrix kernel
# FIRST DIM OF BLOCK MUST BE gv.N_THREADSX_SQRNORM because :
# gv.N_THREADSX_SQRNORM and gv.N_FEATURES ar passed as constants
# because they are used to define size of shared memory

# Since this kernel uses block or 2D threads and
# we have the limitation of 1024 threads per block
# also if block=(gv.N_THREADSX_SQRNORM, Nthreadsy)
# we must have Nthreadsy*gv.N_THREADSX_SQRNORM <= 1024
consts_sqrnorm = {'N_COLUMNS': gv.N_FEATURES,
                  'N_THREADS_X': gv.N_THREADSX_SQRNORM}
sqr_norm = Kernel.Kernel("SqrNormRowsKernel",
                         kernel_dic["SqrNormRowsKernel"],
                         consts_sqrnorm)
sqr_norm.compile()


# Select and return copy of the submatrix of movies id and ratings
# corresponding the movies rated by a given user
rated_by_usr = Kernel.Kernel("RatedByUsrKernel",
                             kernel_dic["RatedByUsrKernel"])
rated_by_usr.compile()


# Select and return copy of the submatrix of users id and ratings
# corresponding to the the rating base of a given movie
rating_usrs = Kernel.Kernel("MviRatingUsrsKernel",
                            kernel_dic["MviRatingUsrsKernel"])
rating_usrs.compile()


# Select and return a copy of the sub matrix of features
# which column correspond either to the movies seen by a given
# user of the users that have rated a given movie
subset_sel = Kernel.Kernel("SubsetSelKernel",
                           kernel_dic["SubsetSelKernel"])
subset_sel.compile()


# Compute the vector of squared error for each row of the ratings matrix
sqr_err_vect = Kernel.Kernel("SqrErrVectKernel",
                             kernel_dic["SqrErrVectKernel"])
sqr_err_vect.compile()


# Compute the vector of penalization along the ratings matrix
pen_vect = Kernel.Kernel("PenVectKernel",
                         kernel_dic["PenVectKernel"])
pen_vect.compile()
