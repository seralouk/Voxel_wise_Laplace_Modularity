# -----------------------------------------------------------
#
#    Custom functions for fast modularity/laplacian workflow
#
#    Created by:         Serafeim Loukas 
#    Last checked:       19.02.2020
#   
# -----------------------------------------------------------

import numpy as np


def standardize_TCS_fast(X, modify_inplace=True):
	"""
	This function standardizes the voxel-wise time-courses stored as rows in X.
	Created by: Loukas Serafeim, Apr 2020, v1

	Parameters
	----------
	X: numpy array 
		Containing the TCS with shape [n_ROIs, Time]

	modify_inplace: boolean, optional (default=True)
		Whether to modify the input matrix inplace
	
	Returns
	----------
	X: numpy array
		The standardized (z-scored) TCSs with shape [n_ROIs, Time]
	"""
	if not modify_inplace:
		X = X.copy()               
	# Note: A loop is prefered because scipy.stats.zscore will crush for large dimensions                            
	for line in range(X.shape[0]): 
		X[line,:] = (X[line,:] - np.mean(X[line,:])) / (np.std(X[line,:], ddof=1))
	
	return X


def build_Xu_Xl(X, u_th, l_th):
	"""
	This function builds Xu and Xl matrices.
	Created by: Loukas Serafeim, Apr 2020, v1

	Parameters
	----------
	X: numpy array 
		It contains the TCS with shape [n_ROIs, Time]
	
	u_th: scalar
		the upper threshold
	
	l_th: scalar
		the lower threshold

	Returns
	----------
	Xu, Xl: numpy matrices 
		These arrays have the same shape as X i.e. [n_ROIs, Time]
	"""
	#* Build Xu and Xl matrices
	Xu = np.zeros(X.shape)
	Xl = np.zeros(X.shape)

	#* raw values thresholding
	Xu[X >= u_th] = X[X >= u_th] # X[X >= u_th] if X[X >= u_th] > u otherwise 0
	Xl[X <= l_th] = X[X <= l_th] # X[X <= l_th] if X[X <= l_th] > u otherwise 0
	
	return Xu, Xl



