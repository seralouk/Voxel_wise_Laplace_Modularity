# -------------------------------------------------------
#
#    main - function to execute fast laplacian workflow
#    ACCORDANCE
#
#    Created by:         Serafeim Loukas 
#    Last checked:       11.03.2020
#
# ------------------------------------------------------

import numpy as np, scipy, os, nibabel as nib, pandas as pd
from scipy.sparse.linalg import eigs,eigsh, LinearOperator
from scipy import stats
from Utilities.utils import standardize_TCS_fast, build_Xu_Xl
from zipfile import ZipFile

#* Load the GM extracted voxel-wise time-courses of a signle subject (and single run)
print("\nCase: Laplacian using Accordance as adjacency\n")
print(' '*1+"Loading the GM voxel-wise time-courses of a single subject\n")

zip_file = ZipFile("./Data/GM_voxel_wise_subj_104416_run_1.csv.zip")
data = pd.read_csv(zip_file.open("GM_voxel_wise_subj_104416_run_1.csv"), header=None, low_memory=False, na_filter=False)
X = data.values.T
print(' '*2+"Initial dimensions of X matrix is: {} i.e. (# voxels, # time-points).\n".format(X.shape))

N_voxels = X.shape[0]                                # number of voxels
T = X.shape[1]                                       # number of time-points
u_th, l_th = 0.0 , 0.0                               # the upper and lower thresholds
k = 5                                                # how many eigenvals and eigenvecs to estimate

#* Standardize each voxel timecourse to 0 meand and 1 std
X = standardize_TCS_fast(X, modify_inplace=True)     # z-score each row (voxel) of X

#* Build Xu and Xl matrices
Xu, Xl = build_Xu_Xl(X, u_th, l_th)                  # Build Xu and Xl matrices
# del(X)                                             # be smart - free up some memory

# Get number of voxels and time-points
T = Xu.shape[1]                                      # time-course length
N_voxels = Xu.shape[0]                               # number of voxels


#* Define the main function
def Binline_L_with_Accordance(x):
	"""
	This **inline** function computes the Laplacian decomposition using 
	Accordance matrix as adjacency in a fast way, without building any 
	matrix **explicitly**. The operation are performed from right to left.

	Created by: Loukas Serafeim, Apr 2020, v1

	Parameters
	----------
		None, but the matrix X containing the TCS with shape [n_ROIs, Time] should be defined **in advance.**
		`x` is representing an eigenvector (will be estimated in a fast way).

	"""
	d1 = np.dot( Xu.T, np.ones((N_voxels,1)) )
	d1 = Xu.dot(d1) / (T-1.0)
	d2 = np.dot( Xl.T, np.ones((N_voxels,1)) )
	d2 = Xl.dot(d2) / (T-1.0)
	d = d1 + d2
	# Correct the degrees. Reminder: Acc has ones on the diagonal by definition.
	d = d - 1.0
	d_inv = 1. / np.sqrt(d)

	# build the " d^(-0.5) x" part
	d_inv_x = np.multiply(d_inv.ravel(), x)

	# build the " A d^(-0.5) x" part
	c1 = np.dot(Xu.T, d_inv_x)
	c1 = Xu.dot(c1) / (T-1.0)
	c2 = np.dot(Xl.T, d_inv_x)
	c2 = Xl.dot(c2) / (T-1.0)
	C_d_x = c1 + c2

	# build the d^(-0.5) A d^(-0.5) x
	D_inv_C_D_inv_x = np.multiply(d_inv.ravel(), C_d_x)

	# build the "I x = x" part
	I_x = x.copy()
	y = I_x - D_inv_C_D_inv_x
	return y


#* Pass the above inline function into a Linear Operator
Lin_oper = LinearOperator(shape=(Xu.shape[0], Xu.shape[0]) , matvec =  Binline_L_with_Accordance)

#* Get a subset of the spectrum of matrix L = I - D^(-0.5) A D^(-0.5)
print(' '*3+"Computing the first {} smallest eigvals and eigenvecs of L\n".format(k))
print(' '*4+"Please wait (ETA: <1 min)\n")
vals, vecs = eigsh(Lin_oper, k=k, which='SA', ncv=2*k, maxiter=2500, tol=1e-10)
print(' '*5+"Estimation of the subset of the spectrum is done\n")

#* Sort the estimated eigenvalues and eigenvectors
idx = vals.argsort() # order from smallest to largest
vals, vecs = vals[idx], vecs[:,idx]

#* Note: there is a shift in the eigenvalues: lambda_true = vals + 1/d
#  Proof: In "Binline_L_with_Accordance" we have corrected the degrees for the fact that Accordance has 1s on the
#  diagonal by definition. However, the construction of the adjacency still includes these 1s i.e. A=A+cI with c=1
#  Lx = Ix - D^(-0.5)(A+cI)D^(-0.5)x = Ix - D^(-0.5) A D^(-0.5)x - D^(-0.5)(cI)D^(-0.5)x = (lambda_true - D^(-1))
#  So: lambda_true = vals + 1/d
#  The eigenvectors are invariant

#* Save spectrum on the disk
# save_to_ ='...' # define path 
# np.save(save_to_ + "GROUP_vecs_{}_laplace_acc".format(k),vecs)
# np.save(save_to_ + "GROUP_vals_{}_laplace_acc".format(k),vals)

#* Map the estimated laplacian eigenvectors on the brain
#* using the GM mask that was used to extract the voxel-wise time-courses.
print(' '*6+"Mapping eigenvector to a brain map\n")
GMmask = nib.load('./Data/GM_mask.nii')
GMmask_data = GMmask.get_fdata().astype(bool)
GMmask_flat = GMmask_data.reshape(1,-1, order='F').copy()   # F order in this case because the extracted voxel-wise 
                                                            # time-courses were based on F order flattening.
#* The Fiedler laplacian eigenvector
signal = vecs[:,1].copy()                                   # create a copy

#* Map the leading laplacian eigenvector and Reshape into a brain shape
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
mynii_name = "leading_eigenv_laplace_acc.nii"                      # do not forget the file extension
mynii = nib.Nifti1Image(mapped_signal, GMmask.affine, GMmask.header)
nib.save(mynii, save_nii_to + mynii_name + '.gz')          # save the eigenmode as a brain map in nii format
print(' '*7+"All done. The results have been saved in: {}\n".format(save_nii_to))




