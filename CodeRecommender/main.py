
# Third party imports
from inspect import getsourcefile
from os.path import abspath
import numpy as np
import matplotlib.pyplot as plt
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import math
import time
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

# Local imports
import kernels
import cpu_version
import initialization
import data_treatment as data
import global_variables as gv
import als_gpu
import cost_and_errors


################DATA#SECTION##############################################
##########################################################################

# Find abs path to dataset in default configuration
if not gv.PATH:
    path = abspath(getsourcefile(lambda: 0))
    path = str(path)
    cutted = path.split("CodeRecommender", 1)[0]
    if gv.DATASET_SIZE == "1M":
        datapath = cutted + "Datasets/ratings1M.csv"
        # Load the dataset
        dataset = data.load_dataset(datapath)
    else:
        datapath = cutted + "Datasets/ratings100K.csv"
        # Load the dataset
        dataset = data.load_dataset(datapath)

else:
    # Load the dataset
    dataset = data.load_dataset(gv.PATH)


# Multiply ratings by 10 which enables
# us to cast whole dataset as ints
# more efficient memorywise than storing
#ints in floats
dataset = data.dataset_to_ints(dataset)


# Divide the dataset in learn and test sets
learnfrac = 0.7
learn, test = data.learn_test_div(dataset)


# Drop timestamp column which is unused
learn.drop(["timestamp"], axis=1, inplace=True)
test.drop(["timestamp"], axis=1, inplace=True)


# Keep only users and movies common between
#learn and test
learn, test = data.only_common(learn, test)


# Add auxiliary columns "userBijId" and "movieBijId"
# for indexing users and movies resp. on range (0, nusers), (0, nmovies)
learn = data.reindex_usrs_mvies(learn)
test = data.reindex_usrs_mvies(test)


# The following arrays will be important in keeping track
# of indexes in the flatten array setting of cuda C kernels
# As a toy example, in a setting where user 0 has seen 5 movies,
# user 1 has seen 8 and user 2 has seen 1, nseen_mvis = [5, 8, 1]
nrated_usr = np.array(
    data.get_cols_nratings(
        learn,
        "userBijId"),
    dtype=np.uint32)
cum_nrated_usr = np.cumsum(nrated_usr).astype(np.uint32)
# As a toy example, in a setting where movie 0 has 40 ratings,
# movie 1 has 20 and movie 2 has 60, nrating_user = [40, 20, 60]
nratings_mvi = np.array(
    data.get_cols_nratings(
        learn,
        "movieBijId"),
    dtype=np.uint32)
cum_nratings_mvi = np.cumsum(nratings_mvi).astype(np.uint32)


# Having two versions may seem unoptimal
# but this will prove highly important
# to avoid comings and goings from cpu to gpu during training

# Sorted by users versions
learn_usr = learn.sort_values(by=["userBijId"])
test_usr = test.sort_values(by=["userBijId"])

# Sorted by movies versions
learn_mvi = learn.sort_values(by=["movieBijId"])
test_mvi = test.sort_values(by=["movieBijId"])


# Arrays of ratings
# uint (all positive) single precision for gpu

# usr sorted
learn_usr_mat = data.dataset_asarray(learn_usr, np.uint32)
test_usr_mat = data.dataset_asarray(test_usr, np.uint32)

# mvi sorted
learn_mvi_mat = data.dataset_asarray(learn_mvi, np.uint32)
test_mvi_mat = data.dataset_asarray(test_mvi, np.uint32)


# Get the parameters from learn database
Nu = data.get_nusrs(learn)
Nm = data.get_nmvis(learn)
Nr = data.get_nrtgs(learn)
Nrt = data.get_nrtgs(test)
Nf = gv.N_FEATURES

lamb = gv.LAMBDA


################GPU#INITIALIZATIONS#SECTION###############################
##########################################################################

# Nm as a (1, 1) gpuarray
Nm_gpu = initialization.scalar_to_1d_gpuarray(Nm, np.uint32)
# Nu as a (1, 1) gpuarray
Nu_gpu = initialization.scalar_to_1d_gpuarray(Nu, np.uint32)
# Nr as a (1, 1) gpuarray
Nr_gpu = initialization.scalar_to_1d_gpuarray(Nr, np.uint32)

Nf_gpu = initialization.scalar_to_1d_gpuarray(Nf, np.uint32)

Nrt_gpu = initialization.scalar_to_1d_gpuarray(Nrt, np.uint32)

# initialize user features matrix on gpu
p_gpu, p_cpu = initialization.init_features_mat(Nu, Nf, np.float32)

# initialize movies features matrix on gpu
q_gpu, q_cpu = initialization.init_features_mat(Nm, Nf, np.float32)

# Ratings matrix sorted by user on the gpu
R_gpu_usr = gpuarray.to_gpu(learn_usr_mat)

# Ratings matrix sorted by movies on the gpu
R_gpu_mvi = gpuarray.to_gpu(learn_mvi_mat)

# Test set on gpu, order does not matter
R_gpu_test = gpuarray.to_gpu(test_usr_mat)

# Transfer to gpu of the previously defined following arrays:
nrated_usr_gpu = gpuarray.to_gpu(nrated_usr)
cum_nrated_usr_gpu = gpuarray.to_gpu(cum_nrated_usr)
nratings_mvi_gpu = gpuarray.to_gpu(nratings_mvi)
cum_nratings_mvi_gpu = gpuarray.to_gpu(cum_nratings_mvi)


############ALS#CPU########################################################
###########################################################################

if gv.RUN_CPU:
    # Get params for exec from global_variables.py
    maxit = gv.MAXIT
    stop = gv.STOP

    print("Running ALS optimization on CPU")
    print(" ")

    # Start processor time clock (on unix, on windows will take the wall clock)
    start_cpu = time.clock()

    # Perform ALS on cpu
    costs_cpu, msesl_cpu, msest_cpu = cpu_version.iter_solve_update(learn_usr_mat,
                                                                    learn_mvi_mat,
                                                                    test_usr_mat,
                                                                    p_cpu,
                                                                    q_cpu,
                                                                    nrated_usr,
                                                                    nratings_mvi,
                                                                    lamb,
                                                                    maxit,
                                                                    stop)

    # End clock
    end_cpu = time.clock()

    # Exec time
    elapsed_cpu = end_cpu - start_cpu

    # Print exec time
    print(" ")
    print(" ")
    print("Elapsed time for CPU execution:")
    print(elapsed_cpu)
    print(" ")
    print(" ")

    # Apply square root function to mean square errors
    rmsesl_cpu = [math.sqrt(i) for i in msesl_cpu]
    rmsest_cpu = [math.sqrt(i) for i in msest_cpu]

    # Print costs throughout iterations
    print("Costs CPU")
    print(costs_cpu)
    print(" ")
    print(" ")

    # Print rmse on learn set throughout iterations
    print("RMSE learn CPU")
    print(rmsesl_cpu)
    print(" ")
    print(" ")

    # Print rmse on test set throughout iterations
    print('RMSE test CPU')
    print(rmsest_cpu)
    print(" ")
    print(" ")


###############ALS#GPU####################################################
##########################################################################

# Get params for exec from global_variables.py
maxit = gv.MAXIT
stop = gv.STOP

print("Running ALS optimization on GPU")
print(" ")
print(" ")

# Start processor time clock (on unix, on windows will take the wall clock)
start_gpu = time.clock()

# Perform ALS on gpu
costs_gpu, msesl_gpu, msest_gpu = als_gpu.iterate_als(R_gpu_usr,
                                                      R_gpu_mvi,
                                                      R_gpu_test,
                                                      p_gpu,
                                                      q_gpu,
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
                                                      stop)

# End of clock
end_gpu = time.clock()

# gpu exec time
elapsed_gpu = end_gpu - start_gpu

# print gpu exec time
print("Elapsed time for GPU execution:")
print(elapsed_gpu)
print(" ")
print(" ")

# Apply square root function to mean square errors
rmsesl_gpu = [math.sqrt(i) for i in msesl_gpu]
rmsest_gpu = [math.sqrt(i) for i in msest_gpu]

# Print costs throughout iterations
print("Costs GPU")
print(costs_gpu)
print(" ")
print(" ")

# Print rmse on learn set throughout iterations
print("RMSE learn GPU")
print(rmsesl_gpu)
print(" ")
print(" ")

# Print rmse on test set throughout iterations
print('RMSE test GPU')
print(rmsest_gpu)
print(" ")
print(" ")


#######PLOTS#AND#OUTPUTS#####################################################
#############################################################################


if gv.RUN_CPU:
    # Plot of costs and rmses
    fig, axes = plt.subplots(2, sharex=True)
    plt.xlabel("Iteration")
    fig.suptitle("CPU")
    axes[0].plot(costs_cpu, marker="o")
    axes[0].set_ylabel("Cost")
    axes[1].plot(rmsesl_cpu, marker="o", label="Learn")
    axes[1].plot(rmsest_cpu, marker="o", label="Test")
    axes[1].set_ylabel("RMSE")
    plt.legend()
    fig.show()

    # Zoom on the plot of rmse after the first iteration
    plt.figure()
    plt.title("CPU : Zoom on RMSE after 1st iteration")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.plot(list(range(1, len(rmsest_cpu))),
             rmsesl_cpu[1:], marker="o", label="Learn")
    plt.plot(list(range(1, len(rmsest_cpu))),
             rmsest_cpu[1:], marker="o", label="Test")
    plt.legend()
    plt.show()


# Plot of costs and rmses
fig, axes = plt.subplots(2, sharex=True)
plt.xlabel("Iteration")
fig.suptitle("10M dataset - GPU")
axes[0].plot(costs_gpu, marker="o")
axes[0].set_ylabel("Cost")
axes[1].plot(rmsesl_gpu, marker="o", label="Learn")
axes[1].plot(rmsest_gpu, marker="o", label="Test")
axes[1].set_ylabel("RMSE")
plt.legend()
plt.show()


# Zoom on the plot of rmse after the first iteration
plt.figure()
plt.title("10M dataset - GPU : Zoom after 1st iteration")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.plot(list(range(1, len(rmsest_gpu))),
         rmsesl_gpu[1:], marker="o", label="Learn")
plt.plot(list(range(1, len(rmsest_gpu))),
         rmsest_gpu[1:], marker="o", label="Test")
plt.legend()
plt.show()


# Save output for gpu only
if gv.SAVE_OUTPUT:
    errtab = pd.DataFrame(columns=["cost", "rmse_learn", "rmse_test"])
    errtab["cost"] = costs_gpu
    errtab["rmse_learn"] = rmsesl_gpu
    errtab["rmse_test"] = rmsest_gpu
    errtab.to_csv(gv.OUTPUT_PATH + "costs_and_errs.csv")
    usr_features = pd.DataFrame(data=p_gpu.get())
    usr_features.to_csv(gv.OUTPUT_PATH + "usr_feats.csv")
    mvi_features = pd.DataFrame(data=q_gpu.get())
    mvi_features.to_csv(gv.OUTPUT_PATH + "mvi_feats.csv")
