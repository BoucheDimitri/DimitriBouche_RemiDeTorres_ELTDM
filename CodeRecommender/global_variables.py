

#Parameters for execution

#Dataset to use if dataset from folder Datasets is used Either "1M" or "100K"
DATASET_SIZE = "100K"
#Path to dataset to use other dataset
PATH = None
#Should outputs be saved on computer ?
SAVE_OUTPUT = False
#If SAVE_OUTPUT, where should they be saved ?
OUTPUT_PATH = "~/Bureau/ENSAE/ELTDM/Outputs/"
#Should CPU version of algorithm should also be run ?
RUN_CPU = True

#Number of features to use for movies and users
N_FEATURES = 4

#Arguments for alternating least squares
#Regularization coefficient
LAMBDA = 0.05
#Maximum number of iterations
MAXIT = 10
#Stopping criterion on the decrease of cost
STOP = 10

#Number of threads to execute sqr norm kernel
#NOT ADVISED TO MODIFY
N_THREADSX_SQRNORM = 32
