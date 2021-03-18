# -------------------------------------------------------
#
#    main - function to execute fast modularity workflow
#    DISCORDANCE
#
#    Created by:         Serafeim Loukas 
#    Last checked:       19.02.2020
#
#   
# ------------------------------------------------------

import numpy as np, scipy, os, nibabel as nib, pandas as pd
from scipy.sparse.linalg import eigs,eigsh, LinearOperator
from scipy import stats
from Utilities.utils import standardize_TCS_fast, build_Xu_Xl
from zipfile import ZipFile


#* Load the GM extracted voxel-wise time-courses of a signle subject (toy example)
print("\nCase: Modularity using Accordance as adjacency\n")
print(' '*1+"Loading the GM voxel-wise time-courses of a single subject\n")

zip_file = ZipFile("./Data/GM_voxel_wise_toy_example.csv.zip")
data = pd.read_csv(zip_file.open("GM_voxel_wise_toy_example.csv"), header=None, low_memory=False, na_filter=False)
X = data.values.T
print(' '*2+"Initial dimensions of X matrix is: {} i.e. (# voxels, # time-points).\n".format(X.shape))

N_voxels = X.shape[0]                                # number of voxels
T = X.shape[1]                                       # number of time-points
u_th, l_th = 0.00, 0.00                              # the upper and lowe thresholds for Accordance / Discordance construction 
k = 5                                                # how many eigenvals and eigenvecs to estimate

#* Standardize each voxel timecourse to 0 meand and 1 std
X = standardize_TCS_fast(X, modify_inplace=True)     # z-score each row (voxel) of X

#* Build Xu and Xl matrices
Xu, Xl = build_Xu_Xl(X, u_th, l_th)    # Build Xu and Xl matrices
# del(X)                                             # be smart - free up some memory

# Get number of voxels and time-points
T = Xu.shape[1]                                      # time-course length
N_voxels = Xu.shape[0]                               # number of voxels


#* Define the main function
def Binline_B_with_Discordance(x):
	"""
	This **inline** function computes the Modularity decomposition using 
	Discordance matrix as adjacency in a fast way, without building any 
	matrix **explicitly**. The operation are performed from right to left.

	Created by: Loukas Serafeim, Apr 2020, v1

	Parameters
	----------
		None, but the matrix X containing the TCS with shape [n_ROIs, Time] should be defined **in advance.**
		`x` is representing an eigenvector (will be estimated in a fast way).

	"""
	c1 = np.dot(Xl.T, x)
	c1 = Xu.dot(c1) / (T-1.0)
	c2 = np.dot(Xu.T, x)
	c2 = Xl.dot(c2) / (T-1.0)
	C = c1 + c2

	d1 = np.dot( Xl.T, np.ones((N_voxels,1)) )
	d1 = Xu.dot(d1) / (T-1.0)
	d2 = np.dot( Xu.T, np.ones((N_voxels,1)) )
	d2 = Xl.dot(d2) / (T-1.0)
	d = d1 + d2
	
	M = np.sum(d)
	dh = np.dot(d.T, (x /(float(M))))
	y = C - np.dot(d,dh)
	return y


#* Pass the above inline function into a Linear Operator
Lin_oper = LinearOperator(shape=(Xu.shape[0], Xu.shape[0]) , matvec =  Binline_B_with_Discordance)

#* Get a subset of the spectrum of matrix B = A - (d*d' / M)
print(' '*3+"Computing the first {} largest eigvals and eigenvecs of B\n".format(k))
print(' '*4+"Please wait (ETA: <1 min)\n")

#* Note: Disc by definition has values in [-1,0], so if we do not flip the sign (as we do in this implementation) we need
# to approximate the largest algebraic. If we flip the sign then the smallest algebrainc (most negatives).
# Both are mathematically equivalent i.e. Bx = lx or -Bx = -lx

vals, vecs = eigsh(Lin_oper, k=k, which='LA', ncv=2*k, maxiter=2500, tol=1e-10)
print(' '*5+"Estimation of the subset of the spectrum is done\n")

#* Sort the estimated eigenvalues and eigenvectors
idx = vals.argsort()[::-1]
vals, vecs = vals[idx], vecs[:,idx]

#* Save spectrum on the disk
# save_to_ ='...' # define path 
# np.save(save_to_ + "GROUP_vecs_{}_acc".format(k),vecs)
# np.save(save_to_ + "GROUP_vals_{}_acc".format(k),vals)

#* Map the estimated modularity eigenvectors on the brain
#* Load the GM mask that was used to extract the voxel-wise time-courses.
print(' '*6+"Mapping eigenvector to a brain map\n")
GMmask = nib.load('./Data/GM_mask.nii')
GMmask_data = GMmask.get_fdata().astype(bool)
GMmask_flat = GMmask_data.reshape(1,-1, order='F').copy()   # F order in this case because the extracted voxel-wise 
                                                            # time-courses were based on F order flattening.
#* The leading modularity eigenvector
signal = vecs[:,0].copy()                                   # create a copy

#* Map the leading modularity eigenvector and Reshape into a brain shape
x,y,z = GMmask_data.shape[0], GMmask_data.shape[1], GMmask_data.shape[2]
unmasked = np.zeros(GMmask_data.shape)
unmasked = unmasked.ravel(order='F')
unmasked[GMmask_flat.ravel(order='F')] = signal
mapped_signal = unmasked.reshape((x,y,z),order='F')

#* Save this 3D (ndarry) numpy using the header information of the mask (i.e. mymask.affine, mymask.header)
save_nii_to = "./Results/"                                  
if not os.path.exists(save_nii_to):
	os.makedirs(save_nii_to)

#* define filename and extension (can be filename.nii or filename.nii.gz)
mynii_name = "leading_eigenv_modularity_disc.nii"                      # do not forget the file extension
mynii = nib.Nifti1Image(mapped_signal, GMmask.affine, GMmask.header)
nib.save(mynii, save_nii_to + mynii_name + '.gz')           # save the eigenmode as a brain map in nii format
print(' '*7+"All done. The results have been saved in: {}\n".format(save_nii_to))

