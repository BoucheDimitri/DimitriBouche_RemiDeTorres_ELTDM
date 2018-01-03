# Movies recommender system on GPU #

This is an implementation of a collaborative filtering algorithm on GPU using Alternated least squares for optimization.

## Getting started ##

For quick start just change dir to ./Code and execute $ python main.py

*Note : two ready to run datasets are in the folder Datasets, the 100K is used by default*

Other parameters are available in the **./Code/global_variables.py files**, to change the dataset, the number of features etc... edit the parameters in the file before running main.py

## Dependencies ##

CUDA toolkit >= 7.0 is mandatory (cusolver library was added in CUDA 7.0)

The project was develpped using Python 2.7

The following packages are mandatory (version on my setup is in parenthesis but other versions should work)

* pandas (0.21.1)
* numpy (1.13.3)
* pycuda (2017.1.1)
* scikit-cuda (0.5.2)
* matplotlib (2.1.1)

## Math wiki ##

See file **report.pdf** for a detailed report on the maths, the implementation and the results.


