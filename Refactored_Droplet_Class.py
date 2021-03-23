#! This will create a Droplet_Analysis Object for a given (SINGLE) input, with all the functions we need: 

import numpy as np
import os, sys

import scipy.io as sp_io
from scipy.integrate import tplquad as sp_int_tpq
from scipy.special import sph_harm
import scipy.stats as sp_stats
import scipy.sparse as sp_sprs

# For vtp plots: #!!!! REFACTOR PLOTTING !!!!#
import read_write
write_vtp_atz = read_write.write_vtp;
read_vtp_atz = read_write.read_vtp;

MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
Output_Plots_Dir = os.path.join(MY_DIR, 'Outputs') # where we put output data
#PARENT_DIR = os.path.abspath(os.path.join(MY_DIR, os.pardir))
#CORE_DIR = os.path.join(PARENT_DIR, 'SPB_CoreCodes')

# BJG: added this here to allow this to get modules below in super-folder:
#sys.path.append("../SPB_CoreCodes")

import numpy as np

import lbdv_info_SPB as lbdv_i 
import sph_func_SPB as sph_f
import manifold_SPB as mnfd
import euc_k_form_SPB as euc_kf
import plots_SPB as plts
import charts_SPB as chrts

import gdist as gd # for graph distances

# Explanation: http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
# Code (below): https://stackoverflow.com/questions/58501545/python-fit-3d-ellipsoid-oblate-prolate-to-3d-points

def ls_ellipsoid(xx,yy,zz):                                  
	#finds best fit ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
	#least squares fit to a 3D-ellipsoid
	#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz  = 1
	#
	# Note that sometimes it is expressed as a solution to
	#  Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz  = 1
	# where the last six terms have a factor of 2 in them
	# This is in anticipation of forming a matrix with the polynomial coefficients.
	# Those terms with factors of 2 are all off diagonal elements.  These contribute
	# two terms when multiplied out (symmetric) so would need to be divided by two

	# change xx from vector of length N to Nx1 matrix so we can use hstack
	x = xx[:,np.newaxis]
	y = yy[:,np.newaxis]
	z = zz[:,np.newaxis]

	#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz = 1
	J = np.hstack((x*x,y*y,z*z,x*y,x*z,y*z, x, y, z))
	K = np.ones_like(x) #column of ones

	#np.hstack performs a loop over all samples and creates
	#a row in J for each x,y,z sample:
	# J[ix,0] = x[ix]*x[ix]
	# J[ix,1] = y[ix]*y[ix]
	# etc.

	JT=J.transpose()
	JTJ = np.dot(JT,J)
	InvJTJ=np.linalg.inv(JTJ);
	ABC= np.dot(InvJTJ, np.dot(JT,K)) #!!!! LOOK AT RESIDUALS TO GET ELLIPSOID ERRORS !!!!#

	# Rearrange, move the 1 to the other side
	#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0
	#    or
	#  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0
	#  where J = -1
	eansa=np.append(ABC,-1)

	return (eansa)

# For above Ellipsoid Code:
def polyToParams3D(vec,printMe):                             
	#gets 3D parameters of an ellipsoid. Found at http://www.juddzone.com/ALGORITHMS/least_squares_3D_ellipsoid.html
	# convert the polynomial form of the 3D-ellipsoid to parameters
	# center, axes, and transformation matrix
	# vec is the vector whose elements are the polynomial
	# coefficients A..J
	# returns (center, axes, rotation matrix)

	#Algebraic form: X.T * Amat * X --> polynomial form

	if printMe: print('\npolynomial\n',vec)

	Amat=np.array(
	[
	[ vec[0],     vec[3]/2.0, vec[4]/2.0, vec[6]/2.0 ],
	[ vec[3]/2.0, vec[1],     vec[5]/2.0, vec[7]/2.0 ],
	[ vec[4]/2.0, vec[5]/2.0, vec[2],     vec[8]/2.0 ],
	[ vec[6]/2.0, vec[7]/2.0, vec[8]/2.0, vec[9]     ]
	])

	if printMe: print('\nAlgebraic form of polynomial\n',Amat)

	#See B.Bartoni, Preprint SMU-HEP-10-14 Multi-dimensional Ellipsoidal Fitting
	# equation 20 for the following method for finding the center
	A3=Amat[0:3,0:3]
	A3inv=np.linalg.inv(A3)
	ofs=vec[6:9]/2.0
	center=-np.dot(A3inv,ofs)
	if printMe: print('\nCenter at:',center)

	# Center the ellipsoid at the origin
	Tofs=np.eye(4)
	Tofs[3,0:3]=center
	R = np.dot(Tofs,np.dot(Amat,Tofs.T))
	if printMe: print('\nAlgebraic form translated to center\n',R,'\n')

	R3=R[0:3,0:3]
	R3test=R3/R3[0,0]
	# print('normed \n',R3test)
	s1=-R[3, 3]
	R3S=R3/s1
	(el,ec)=np.linalg.eig(R3S)

	recip=1.0/np.abs(el)
	axes=np.sqrt(recip)
	if printMe: print('\nAxes are\n',axes  ,'\n')

	inve=np.linalg.inv(ec) #inverse is actually the transpose here
	if printMe: print('\nRotation matrix\n',inve)
	return (center,axes,inve, ec)


# Use level set calculated above to get normals to Ellipsoid, for computing errors (instead of PyCompadre Normals):
def Ellipsoid_Level_Set_Normals(xx,yy,zz, EllipsCoef):                                  
	#least squares fit to a 3D-ellipsoid:
	# F(X) =  Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz - 1 = 0 (LEVEL SET)
	# Ellipsoid Coef = [A,B,C,...]
	# normals given by \grad F(X)/ \| \grad F(X) \|

	A = EllipsCoef.flatten()[0]
	B = EllipsCoef.flatten()[1]	
	C = EllipsCoef.flatten()[2]	
	D = EllipsCoef.flatten()[3]	
	E = EllipsCoef.flatten()[4]	
	F = EllipsCoef.flatten()[5]	
	G = EllipsCoef.flatten()[6]	
	H = EllipsCoef.flatten()[7]	
	I = EllipsCoef.flatten()[8]		

	grad_F_x = 2.*A*xx + D*yy + E*zz + G
	grad_F_y = 2.*B*yy + D*xx + F*zz + H 
	grad_F_z = 2.*C*zz + E*xx + F*yy + I

	grad_F_X = np.hstack(( grad_F_x, grad_F_y, grad_F_z )) 
	Vec_Norms = np.sqrt(np.sum(np.multiply(grad_F_X, grad_F_X), axis = 1)).reshape(len(xx), 1)
	grad_F_X_normed = np.divide(grad_F_X, Vec_Norms)
	
	return grad_F_X_normed

	 
# Convert input R^3 points into Ellipsoidal coors:
def Conv_3D_pts_to_Elliptical_Coors(a0, a1, a2, point_cloud_input, LS_Ellps_Inv_Rot_Mat, LS_Ellps_center):

	num_pts_used = len(point_cloud_input)
	U_coors_calc = np.zeros(( num_pts_used, 1 ))
	V_coors_calc = np.zeros(( num_pts_used, 1 ))
		
	for pt_numb in range(num_pts_used):
		y_tilde_pt = np.linalg.solve(LS_Ellps_Inv_Rot_Mat, point_cloud_input[pt_numb, :].reshape(3,1) - LS_Ellps_center.reshape(3, 1)  ) 

		yt_0 = y_tilde_pt[0,0]
		yt_1 = y_tilde_pt[1,0]
		yt_2 = y_tilde_pt[2,0]	

		U_pt = np.arctan2( yt_1*a0, yt_0*a1 )

		if(U_pt < 0):
			U_pt = U_pt + 2.*np.pi

		U_coors_calc[pt_numb] = U_pt

		cylinder_r = np.sqrt(yt_0**2 + yt_1**2)	# r in cylinderical coors for y_tilde
		cyl_r_exp = np.sqrt( (a0*np.cos(U_pt))**2 + (a1*np.sin(U_pt))**2 )

		V_pt = np.arctan2( cylinder_r*a2, yt_2*cyl_r_exp )

		if(V_pt < 0):
			V_pt = V_pt + 2.*np.pi

		V_coors_calc[pt_numb] = V_pt

	return U_coors_calc, V_coors_calc


# Convert Ellipsoidal Coordinates to R^3 points:
def Conv_Elliptical_Coors_to_3D_pts(U_pts_cloud, V_pts_cloud, LS_Ellps_axes, LS_Ellps_Inv_Rot_Mat, LS_Ellps_center):

	num_pts_used = len(U_pts_cloud)
	X_LS_Ellps_calc_pts = np.zeros(( num_pts_used, 1 )) 
	Y_LS_Ellps_calc_pts = np.zeros(( num_pts_used, 1 )) 
	Z_LS_Ellps_calc_pts = np.zeros(( num_pts_used, 1 )) 

	for pt_test in range(num_pts_used): #pt_test_theta in range(num_test_pts_per_dim):
		theta_tp =  U_pts_cloud[pt_test, 0] #theta_pts[pt_test_theta]
		phi_tp = V_pts_cloud[pt_test, 0] #phi_pts[pt_test_phi]
		
		Test_pt = np.dot(LS_Ellps_Inv_Rot_Mat, np.multiply( np.array([ [np.sin(phi_tp)*np.cos(theta_tp)], [np.sin(phi_tp)*np.sin(theta_tp)], [np.cos(phi_tp)] ]), LS_Ellps_axes.reshape(3, 1) )  ) + LS_Ellps_center.reshape(3, 1) 
		
		X_LS_Ellps_calc_pts[pt_test, 0] = Test_pt[0, 0]
		Y_LS_Ellps_calc_pts[pt_test, 0] = Test_pt[1, 0]
		Z_LS_Ellps_calc_pts[pt_test, 0] = Test_pt[2, 0]
	
	return X_LS_Ellps_calc_pts, Y_LS_Ellps_calc_pts, Z_LS_Ellps_calc_pts


# Compute H on Ellipsoid, using formula from worlfram alpha: https://mathworld.wolfram.com/Ellipsoid.html
def Mean_Curvs_on_Ellipsoid(a0, a1, a2, U_pts_calc, V_pts_calc):

	'''		
	x = a0*sin(V)*cos(U)
	y = a1*sin(V)*sin(U)
	z = a2*cos(V)
	'''

	num_H_ellps = a0*a1*a2* ( 3.*(a0**2 + a1**2) + 2.*(a2**2) + (a0**2 + a1**2 -2.*a2**2)*np.cos(2.*V_pts_calc) - 2.*(a0**2 - a1**2)*np.cos(2.*U_pts_calc)*np.sin(V_pts_calc)**2 )

	den_H_ellps = 8.*( (a0*a1*np.cos(V_pts_calc))**2 + ( a2*np.sin(V_pts_calc) )**2 * ( (a1*np.cos(U_pts_calc))**2 + (a0*np.sin(U_pts_calc))**2 ) )**1.5

	H_ellps_pts = num_H_ellps/den_H_ellps
	return H_ellps_pts


# return least squares harmonic fit to point cloud, given choice of basis and degree:
def Least_Squares_Harmonic_Fit(deg_fit_used, U_fit_coors, V_fit_coors, X_input_fit, Y_input_fit, Z_input_fit, Use_True_Harmonics):

	All_Y_mn_pt_in = []

	for n in range(deg_fit_used+1):
		for m in range(-1*n, n+1):
			Y_mn_coors_in = []
			
			# we can use actual harmonics or our basis:
			if(Use_True_Harmonics == True):
				Y_mn_coors_in = sph_harm(m, n, U_fit_coors, V_fit_coors)
			else:
				Y_mn_coors_in = lbdv_i.Eval_SPH_Basis(m, n, U_fit_coors, V_fit_coors)

			All_Y_mn_pt_in.append(Y_mn_coors_in)

	All_Y_mn_pt_in_mat = np.hstack(( All_Y_mn_pt_in ))
	X_sph_coef_vec = np.linalg.lstsq(All_Y_mn_pt_in_mat, X_input_fit)[0]
	Y_sph_coef_vec = np.linalg.lstsq(All_Y_mn_pt_in_mat, Y_input_fit)[0]
	Z_sph_coef_vec = np.linalg.lstsq(All_Y_mn_pt_in_mat, Z_input_fit)[0]

	return X_sph_coef_vec, Y_sph_coef_vec, Z_sph_coef_vec


#!!!! BJG: REFACTOR WITH ABOVE FN !!!!#
def Least_Squares_Harmonic_Fit_SINGLE(deg_fit_used, U_fit_coors, V_fit_coors, Fn_pts_to_fit, Use_True_Harmonics):
	All_Y_mn_pt_in = [] # Fit SINGLE FN INPUT

	for n in range(deg_fit_used+1):
		for m in range(-1*n, n+1):
			Y_mn_pts_in = []

			# we can use actual harmonics or our basis:
			if(Use_True_Harmonics == True):
				Y_mn_pts_in = sph_harm(m, n, U_fit_coors, V_fit_coors)
			else:
				Y_mn_pts_in = lbdv_i.Eval_SPH_Basis(m, n, U_fit_coors, V_fit_coors)

			All_Y_mn_pt_in.append(Y_mn_pts_in)

	All_Y_mn_pt_in_mat = np.hstack(( All_Y_mn_pt_in ))

	SPH_coef_vec = np.linalg.lstsq(All_Y_mn_pt_in_mat, Fn_pts_to_fit)[0]

	return SPH_coef_vec


# bump function weights on pts centered on dist_x_c:
def avg_around_pt(dist_x_c, dists_pts, vals_at_pts, max_dist_used):
	dist_max = max_dist_used/20. #10. #5. #10.
	pts_within_1 = np.where(abs(dists_pts - dist_x_c)<=dist_max, 1., 0.) # only look at points within 1
	#sum_pts_within_1 = np.sum(pts_within_1*vals_at_pts) # we can average over all pts within 1

	num_pts_within_1 = np.sum(pts_within_1) # number of points in bin

	pts_vals_within_1 = np.where(abs(dists_pts - dist_x_c)<=dist_max, vals_at_pts, 0.) # only look at points within 1
	#dists_within_1 = np.where(abs(dists_pts - dist_x_c)<=1., dists_pts, 0.) # only look at points within 1
	weights = np.where( abs(dists_pts - dist_x_c)<=1., np.exp(1.- 1./(dist_max**2 - (dists_pts - dist_x_c)**2 )), 0. )
	sum_weights = np.sum(weights.flatten())
	sum_pts_within_1 = np.sum( np.multiply(pts_vals_within_1, weights).flatten() )

	'''
	#weights =  np.exp(-1.*(pts - x_c)**2/np.log(10.)) 
	weights = np.exp(1.- 1./(1. - (pts_within_1 - x_c)**2 ))

	sum_weights = np.sum(weights.flatten())
	sum_pts_weights = np.sum( np.multiply(vals_at_pts, weights).flatten() )

	print("pts = "+str(pts))
	print("weights = "+str(weights))
	'''

	return sum_pts_within_1, sum_weights #num_pts_within_1


# USE TRIANGULATION TO GET LOCAL MAX/MIN:
def local_min_max_and_dists(field_on_lbdv, dists_lbdv, tris_lbdv):

	quad_fit = len(field_on_lbdv)

	local_max_and_min = np.zeros_like(field_on_lbdv) # should be -1 at local min, +1 at local max, 0 otherwise, 2 if both
	nearest_min_max_dists = np.zeros_like(field_on_lbdv) # 0 if not local max or min; but for local max is distance to nearest local min, vice versa

	nearest_min_max_anisotropic_stress = np.zeros_like(field_on_lbdv)  # 2*(H_in_max - H_in_min), from extrema to nearest partner

	for pt in range(quad_fit):

		H_pt = field_on_lbdv[pt, 0]

		# set to True, change if False:
		pt_is_local_max = True 
		pt_is_local_min = True

		tris_containing_pt = np.where( tris_lbdv == pt )
		#print("tris_containing_pt = "+str(tris_containing_pt))

		rows_pt = tris_containing_pt[0]
		#print("rows_pt = "+str(rows_pt))

		num_occ_pt = np.count_nonzero(tris_lbdv == pt)
		#print("num_occ_pt = "+str(num_occ_pt))

		for row_pt in range(len(rows_pt)):
			row_num_pt = rows_pt[row_pt]
			#print(" row_num_pt = "+str(row_num_pt))
			#print("tris_lbdv[row_num_pt, :] = "+str( tris_lbdv[row_num_pt, :] ))

			# compare to other pts in triangle
			for other_tri_pt_num in range(3):
				other_tri_pt = 	int(tris_lbdv[row_num_pt, other_tri_pt_num])
				#print("other_tri_pt = "+str(other_tri_pt))

				H_other_tri_pt = field_on_lbdv[other_tri_pt, 0]
				
				if(H_other_tri_pt < H_pt):	
					pt_is_local_min = False
				if(H_other_tri_pt > H_pt):	
					pt_is_local_max = False
		
		if(pt_is_local_max == True):
			if(pt_is_local_min == True):
				local_max_and_min[pt] = 2 # local max AND min
			else:
				local_max_and_min[pt] = 1 # local max (only)

		elif(pt_is_local_min == True):
			local_max_and_min[pt] = -1 # local min (only)
		else:
			local_max_and_min[pt] = 0

	num_local_max = np.count_nonzero(local_max_and_min == 1)
	num_local_min = np.count_nonzero(local_max_and_min == -1)
	num_local_both = np.count_nonzero(local_max_and_min == 2)
	#print(str(num_local_max)+" local max, "+str(num_local_min)+" local min, "+str(num_local_both)+" that are both")

	local_max_inds = np.where( local_max_and_min == 1 )[0]
	local_min_inds = np.where( local_max_and_min == -1 )[0]

	#print("local_max_inds = "+str(local_max_inds))

	# list of ALL local min/max pairs' distances and difference in input fields:
	All_local_min_max_pairs_anisotropies = []
	All_local_min_max_pairs_distances = []

	for pt_max in range(num_local_max):
		pt_max_num = local_max_inds[pt_max]
		local_max_field = field_on_lbdv[pt_max_num, 0]		

		for pt_min in range(num_local_min):
			pt_min_num = local_min_inds[pt_min]
			local_min_field = field_on_lbdv[pt_min_num, 0]

			dist_max_min_pt = dists_lbdv[pt_min_num, pt_max_num]
			Anisotropy_max_min_pts = local_max_field - local_min_field
			
			All_local_min_max_pairs_anisotropies.append(Anisotropy_max_min_pts)
			All_local_min_max_pairs_distances.append(dist_max_min_pt)

	All_local_min_max_pairs_anisotropies = np.array(All_local_min_max_pairs_anisotropies, dtype=np.dtype('d'))
	All_local_min_max_pairs_distances = np.array(All_local_min_max_pairs_distances, dtype=np.dtype('d'))

	for pt_max in range(num_local_max):
		pt_max_num = local_max_inds[pt_max]
		#print("pt_max_num = "+str(pt_max_num))

		local_min_dists_to_pt = dists_lbdv[pt_max_num, local_min_inds]
		#print("local_min_dists_to_pt = "+str(local_min_dists_to_pt))

		min_dist_to_local_min = min(local_min_dists_to_pt)
		nearest_min_max_dists[pt_max_num] = min_dist_to_local_min

		#print("min_dist_to_local_min = "+str(min_dist_to_local_min))
		# Calculate 2*(H_max - H_nearest_min):
		ind_in_list_of_nearest_min = np.argwhere(local_min_dists_to_pt == min_dist_to_local_min)
		pt_num_of_nearest_min = local_min_inds[ind_in_list_of_nearest_min][0,0]
		#print("ind_in_list_of_nearest_min = "+str(ind_in_list_of_nearest_min)+", pt_num_of_nearest_min = "+str(pt_num_of_nearest_min))

		nearest_min_max_anisotropic_stress[pt_max_num] = ( field_on_lbdv[pt_max_num, 0] - field_on_lbdv[pt_num_of_nearest_min, 0] )
		#print("nearest_min_max_anisotropic_stress[pt_max_num, 0]  = "+str( nearest_min_max_anisotropic_stress[pt_max_num, 0]  ))

	for pt_min in range(num_local_min):
		pt_min_num = local_min_inds[pt_min]
		#print("pt_mib_num = "+str(pt_min_num))

		local_max_dists_to_pt = dists_lbdv[pt_min_num, local_max_inds]
		#print("local_max_dists_to_pt = "+str(local_max_dists_to_pt))

		max_dist_to_local_min = min(local_max_dists_to_pt)
		nearest_min_max_dists[pt_min_num] = max_dist_to_local_min

		#print("max_dist_to_local_min = "+str(max_dist_to_local_min))
		# Calculate 2*(H_min - H_nearest_max):
		ind_in_list_of_nearest_max = np.argwhere(local_max_dists_to_pt == max_dist_to_local_min)
		pt_num_of_nearest_max = local_max_inds[ind_in_list_of_nearest_max][0,0]
		#print("ind_in_list_of_nearest_max = "+str(ind_in_list_of_nearest_max)+", pt_num_of_nearest_max = "+str(pt_num_of_nearest_max))

		nearest_min_max_anisotropic_stress[pt_min_num] = ( field_on_lbdv[pt_num_of_nearest_max, 0] - field_on_lbdv[pt_min_num, 0] )
		#print("nearest_min_max_anisotropic_stress[pt_min_num, 0]  = "+str( nearest_min_max_anisotropic_stress[pt_min_num, 0] ))

	return local_max_and_min, nearest_min_max_dists, nearest_min_max_anisotropic_stress, All_local_min_max_pairs_distances, All_local_min_max_pairs_anisotropies


# CDF Analysis for distribution:
def CDF_Analysis_of_Data(Input_Data_Field, delta_prob_extrema_exlc):

	sorted_Data_Field = np.sort(Input_Data_Field.flatten()) 

	# from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_histogram.html
	hist_Data_Field = np.histogram(sorted_Data_Field, bins='auto', density=True)
	hist_dist = sp_stats.rv_histogram(hist_Data_Field)

	min_val_excl_Data_Field = hist_dist.ppf(delta_prob_extrema_exlc)
	max_val_excl_Data_Field = hist_dist.ppf(1. - delta_prob_extrema_exlc)

	curv_pts_excluded_Data_Field = np.zeros_like(sorted_Data_Field) # 0: where within .5- \delta of median, -1, where CDF<\delta, 1 where CDF>1-\delta 
	curv_pts_excluded_Data_Field = np.where(sorted_Data_Field < min_val_excl_Data_Field, -1, curv_pts_excluded_Data_Field)
	curv_pts_excluded_Data_Field = np.where(sorted_Data_Field > max_val_excl_Data_Field, 1, curv_pts_excluded_Data_Field)

	return min_val_excl_Data_Field, max_val_excl_Data_Field, curv_pts_excluded_Data_Field, hist_dist


# calculate correlations from input vecs (at same pts), and dists mats:
def Corrs_From_Input_Tris_and_Dists(Input_Vec1, Input_Vec2, dist_mat_input, num_spatial_pts, corr_start_dist_input):

	'''
	Upper_dist_lbdv = np.triu(dist_mat_input, 0).flatten() # we want main diagonal, and below zero'd
	dist_0_args_lbdv = np.sort(np.where(Upper_dist_lbdv <= 0.))[::-1] # we want to find where distance is 0, to delete this
	dists_lbdv_non0 = np.delete(Upper_dist_lbdv, dist_0_args_lbdv)

	Input_vec1_pts = Input_Vec1.reshape(num_spatial_pts, 1)
	Input_vec2_pts = Input_Vec2.reshape(num_spatial_pts, 1)

	Corr_outer_prod_mat_pts = np.dot( Input_vec1_pts, Input_vec2_pts.T )
	Upper_Corr_outer_prod_pts =  np.triu(Corr_outer_prod_mat_pts, 0).flatten()
	Corr_non0_pts = np.delete(Upper_Corr_outer_prod_pts, dist_0_args_lbdv)

	inds_dists_lbdv_sorted = dists_lbdv_non0.argsort()
	Corr_non0_pts_sorted = Corr_non0_pts[inds_dists_lbdv_sorted]
	dists_lbdv_non0_sorted = np.sort(dists_lbdv_non0)

	max_dist_lbdv = int(np.floor(dists_lbdv_non0_sorted.flatten()[-1]))
	'''

	dists_lbdv_non0 = dist_mat_input[np.triu_indices(num_spatial_pts)]
	max_dist_lbdv = int(np.floor(max(dists_lbdv_non0)))

	Corr_outer_prod_mat_pts = np.dot(Input_Vec1.reshape(num_spatial_pts, 1), Input_Vec2.reshape(num_spatial_pts, 1).T )
	Corr_non0_pts = Corr_outer_prod_mat_pts[np.triu_indices(num_spatial_pts)]

	Self_corrs_diag = np.diag(Corr_outer_prod_mat_pts)
	auto_corr_norm = np.average(Self_corrs_diag.flatten(), axis=0) # norm, so auto-corrs are 1 at \delta (= |x - x'|) = 0	

	'''
	print("dist_mat_input = "+str(dist_mat_input))
	print("dists_lbdv_non0 = "+str(dists_lbdv_non0))
	print("Corr_outer_prod_mat_pts = "+str(Corr_outer_prod_mat_pts))
	print("Corr_non0_pts = "+str(Corr_non0_pts))
	print("Self_corrs_diag = "+str(Self_corrs_diag))
	sys.exit()
	'''

	avg_mean_curv_auto_corrs = []
	dists_used = []
	corr_start_dist = int(corr_start_dist_input) # where we START LOOKING AT AUTO-CORRS (Since there is a huge spike at 0)

	#print("dists_lbdv_non0_sorted = "+str(dists_lbdv_non0_sorted))
	#print("Corr_non0_pts_sorted = "+str(Corr_non0_pts_sorted))

	for dist_i in range(corr_start_dist,max_dist_lbdv+1):
		#mean_curv_corr_d_i = bump_fn_around_x_avg(dist_i, dists_lbdv_non0_sorted, Corr_non0_pts_sorted)
		sum_mean_curv_corr_d_i, num_mean_curv_corr_d_i = avg_around_pt(dist_i, dists_lbdv_non0, Corr_non0_pts, max_dist_lbdv)

		if(num_mean_curv_corr_d_i > 0):
			mean_curv_corr_d_i = sum_mean_curv_corr_d_i/num_mean_curv_corr_d_i # average in bin
			if(abs(mean_curv_corr_d_i)/auto_corr_norm <= 1.): # include if corr <= 1
				avg_mean_curv_auto_corrs.append(mean_curv_corr_d_i)
				dists_used.append(dist_i)
			'''
			else: #if(abs(mean_curv_corr_d_i/auto_corr_norm) >= 1.):
				print("for dist = "+str(dist_i)+", mean_curv_corr_d_i = "+str(mean_curv_corr_d_i)+", num_mean_curv_corr_d_i = "+str(num_mean_curv_corr_d_i)+", normed corr_i = "+str( mean_curv_corr_d_i/auto_corr_norm ))
			'''
		'''
		if(dist_i == 1):
			print("\n"+"dist_i = "+str(dist_i))
			print("mean_curv_corr_d_i = "+str(mean_curv_corr_d_i))
			print("avg_mean_curv_auto_corrs = "+str(avg_mean_curv_auto_corrs))
			print("dists_used = "+str(dists_used))
			print("\n")
			sys.exit()
		'''

	num_dists_used = len(dists_used)
	num_corrs_used = len(avg_mean_curv_auto_corrs)

	if(num_dists_used != num_corrs_used):
		print("Error: num_corrs_used = "+str(num_corrs_used)+", num_dists_used = "+str(num_dists_used))
		sys.exit()

	# We get (geodesic) distances we calculate correlations on, using bump fn averages around these distances: 
	auto_corrs_microns_dists = np.array( dists_used, dtype=np.dtype('d')).reshape(num_dists_used, 1)
	auto_corrs_avg = np.array( avg_mean_curv_auto_corrs, dtype=np.dtype('d')).reshape(num_dists_used, 1)
	auto_corrs_avg_normed = auto_corrs_avg / auto_corr_norm

	'''
	print("auto_corrs_microns_dists = "+str(auto_corrs_microns_dists))
	print("auto_corrs_avg = "+str(auto_corrs_avg))
	print("auto_corr_norm = "+str(auto_corr_norm))
	print("auto_corrs_avg_normed = "+str(auto_corrs_avg_normed))
	sys.exit()
	'''

	return corr_start_dist, auto_corrs_microns_dists, auto_corrs_avg, auto_corr_norm, auto_corrs_avg_normed


# geodesic (Great Circle) distance on unit sphere, from haversine formula:
# From: https://en.wikipedia.org/wiki/Haversine_formula
def Haversine_dists_S2_lbdv(deg_lbdv, num_lbdv_pts):

	#num_lbdv_pts = LBDV_Input.lbdv_quad_pts
	LBDV_Input = lbdv_i.lbdv_info(deg_lbdv, num_lbdv_pts)

	dists = np.zeros(( num_lbdv_pts, num_lbdv_pts ))

	for pt_1 in range(num_lbdv_pts):
		theta_1 = LBDV_Input.theta_pts[pt_1, 0]
		phi_1 = LBDV_Input.phi_pts[pt_1, 0]
		lat_1 = np.pi - phi_1 # latitude

		for pt_2 in range(pt_1+1, num_lbdv_pts): # dist with self is 0
			theta_2 = LBDV_Input.theta_pts[pt_2, 0]
			phi_2 = LBDV_Input.phi_pts[pt_2, 0]
			lat_2 = np.pi - phi_2 # latitude
			
			lat_diff = lat_2 - lat_1
			long_diff = theta_2 - theta_1			
			
			h = np.sin(lat_diff/2.)**2 + np.cos(lat_1)*np.cos(lat_2)*(np.sin(long_diff**2/2.)**2)
			d_12 = np.arctan2(np.sqrt(h),np.sqrt(1. -h))
			
			dists[pt_1, pt_2] = d_12
			dists[pt_2, pt_1] = d_12

	return dists


# Create class for SINGLE analyzed Droplet Point Cloud:
class Droplet_Analysis(object):

	def __init__(self, Droplet_Input_Dict):

		self.Drop_Info_Dict = {} # create dictionary to centralize info storage

		# Load specified .mat file, read in point cloud and Elijah's calculation of mean curvature:
		self.MatFile_filepath = Droplet_Input_Dict['MatFile_filepath'] # relative path to folder with .mat files
		self.MatFile_name = Droplet_Input_Dict['MatFile_name'] # name of .mat file (WITH extension)
		mat_file_path = os.path.join(self.MatFile_filepath, self.MatFile_name) #os.path.join(MAT_Files_Dir, mat_file_name)

		mat_contents = sp_io.loadmat(mat_file_path)

		# If we have input .mat file of type 'XYZ_H_and_pixelSize'
		if 'data' in mat_contents.keys(): # 'px_sz_um' 
			self.pixel_size_microns = mat_contents['data']['px_sz_um'][0, 0][0, 0] # INPUT HERE in this case, ignores input from batch script
			self.mean_curvs_input = mat_contents['data']['H_inv_um'][0, 0]
			self.point_cloud_input = mat_contents['data']['XYZ_um'][0, 0]

			#!!!! For NEHA ANALYSIS !!!!#
			if(Droplet_Input_Dict['Neha_Radial_analysis']):
				#print("mat_contents['data']['drop_center_XYZ_orig_px'][0, 0] = "+str( mat_contents['data']['drop_center_XYZ_orig_px'][0, 0] ))
				#print("mat_contents['data']['EK_center_XYZ_orig_px'][0, 0] = "+str( mat_contents['data']['EK_center_XYZ_orig_px'][0, 0] ))
				#print("mat_contents['data']['drop_center_XYZ_um'][0, 0] = "+str( mat_contents['data']['drop_center_XYZ_um'][0, 0] ))
				#print("mat_contents['data']['EK_center_XYZ_um'][0, 0] = "+str( mat_contents['data']['EK_center_XYZ_um'][0, 0] ))
				self.Neha_Rad_Vec = mat_contents['data']['drop_center_XYZ_um'][0, 0] - mat_contents['data']['EK_center_XYZ_um'][0, 0]
				self.len_Neha_Rad_Vec = np.linalg.norm(self.Neha_Rad_Vec, 2)
				
				# examine and plot point closest to EK:
				#neha_vec_on_coords = np.sum(np.multiply(self.point_cloud_input, -1.*self.Neha_Rad_Vec.reshape(1, 3) ), axis=1)
				#index_closest_EK = np.argmax(neha_vec_on_coords)
				self.Closest_EK_pt_vec = np.zeros_like( self.point_cloud_input )
				#self.Closest_EK_pt_vec[index_closest_EK, :] = -1.*self.Neha_Rad_Vec.flatten()
				#print("self.Closest_EK_pt_vec[index_closest_EK, :] = "+str( self.Closest_EK_pt_vec[index_closest_EK, :] ))
				self.Closest_EK_pt_vec[:, 0] = -1.*self.Neha_Rad_Vec.flatten()[0]
				self.Closest_EK_pt_vec[:, 1] = -1.*self.Neha_Rad_Vec.flatten()[1]
				self.Closest_EK_pt_vec[:, 2] = -1.*self.Neha_Rad_Vec.flatten()[2]
				self.Neha_Analysis = True
				
		# If we have input .mat file of type 'justCoordinatesAndCurvatures00...'
		else:
			self.mean_curvs_input = mat_contents['meanCurvatureArray']
			self.point_cloud_input = mat_contents['pointCloudArray']
		
			# We input physical values from experiments, so we can put units on outputs:
			self.pixel_size_microns = Droplet_Input_Dict['Pixel_Size_Microns'] # 1., by default, number of microns per pixel length, so we can multiply by input {X, Y, Z}

			# ADJUST Inputs, to Microns:
			self.point_cloud_input = self.point_cloud_input*self.pixel_size_microns # lengths are in microns
			self.mean_curvs_input = self.mean_curvs_input/self.pixel_size_microns # curvatures are now in 1/microns

		# Tension: \gamma  =.5, by default, so we multiply by this to get anistropic tension for curvature	
		self.TwoGammaVal = 2.*Droplet_Input_Dict['Tension_gamma_val']

		# Plotting File_Path, create sub-dir for this if it doesnt exist:
		self.Output_Plot_Subfolder_Name = Droplet_Input_Dict['Output_Plot_Subfolder_Name'] # subfolder we put this plot it (to group sequences of inputs)
		self.Output_Plot_Subfolder_path = os.path.join(Output_Plots_Dir, self.Output_Plot_Subfolder_Name)

		if not os.path.exists(self.Output_Plot_Subfolder_path):
			os.makedirs(self.Output_Plot_Subfolder_path)

		# for 1st vtu plot:
		Input_Plot_name = "Plot_Inputs_"+str(self.MatFile_name).replace("mat", "vtp") # output plotting
		self.Input_Plot_file_path = os.path.join(self.Output_Plot_Subfolder_path, Input_Plot_name)

		# for 2nd vtu plot:
		Least_Square_Ellps_Plot_name = "Least_Sq_Ellpsoid_"+str(self.MatFile_name).replace("mat", "vtp") # output plotting
		self.Least_Square_Ellps_Plot_file_path = os.path.join(self.Output_Plot_Subfolder_path, Least_Square_Ellps_Plot_name)

		# for 3rd vtu plot:
		SPH_fit_UV_Plot_name = "SPH_fit_UV_"+str(self.MatFile_name).replace("mat", "vtp") # output plotting
		self.SPH_UV_fit_Plot_file_path = os.path.join(self.Output_Plot_Subfolder_path, SPH_fit_UV_Plot_name)

		# for 4th vtu plot:
		SPH_fit_lbdv_Plot_name = "SPH_fit_LBDV_"+str(self.MatFile_name).replace("mat", "vtp") # output plotting
		self.SPH_lbdv_fit_Plot_file_path = os.path.join(self.Output_Plot_Subfolder_path, SPH_fit_lbdv_Plot_name)

		# for 5th vtu plot:
		SPH_fit_lbdv_ellps_Plot_name = "SPH_fit_ELLPS_LBDV_"+str(self.MatFile_name).replace("mat", "vtp") # output plotting
		self.SPH_lbdv_ellps_fit_Plot_file_path = os.path.join(self.Output_Plot_Subfolder_path, SPH_fit_lbdv_ellps_Plot_name)

		# for plotting EK and drop center:
		EK_DC_Plot_name = "Plot_EK_DC_"+str(self.MatFile_name).replace("mat", "vtp") # output plotting
		self.EK_DC_Plot_file_path = os.path.join(self.Output_Plot_Subfolder_path, EK_DC_Plot_name)

		# Get info from point cloud input for calculations we want to do:
		self.H0_avg_input_curvs = np.average(self.mean_curvs_input, axis=0) #1st way we estimate H0: mean of input curvatures Elijah calculated

		self.num_pts = len(self.mean_curvs_input)
		self.avg_position_in = np.average(self.point_cloud_input, axis = 0) 
		self.X_orig_in, self.Y_orig_in, self.Z_orig_in = np.hsplit(self.point_cloud_input, 3)

		self.point_cloud_adj = self.point_cloud_input - self.avg_position_in # CENTER point cloud (for orientation and volume calculations)
		self.X_adj_in, self.Y_adj_in, self.Z_adj_in = np.hsplit(self.point_cloud_adj, 3)


		# We can use PyCompadre to analyze droplet point cloud directly:
		self.Use_PyCompadre = Droplet_Input_Dict['Use_PyCompadre']

		if(self.Use_PyCompadre == True):
			import CalcMeanCurvOfInput as CMI # for esimating curvatures from point cloud (NEEDS PYTHON 3.7 for PyCompadre)
			self.pycompadre_p_order = Droplet_Input_Dict['PyCompadre_p_order'] # It's best to use degree 2
			self.PyComp_Mean_Curvs, self.PyCompadre_Normals_used = CMI.calc_grad_point_cloud(self.point_cloud_adj, self.pycompadre_p_order)

			#2nd way we estimate H0: mean of input curvatures PyCompadre calculated:
			self.H0_avg_PyCompadre_curvs = np.average(self.PyComp_Mean_Curvs, axis=0) 
		
		
		# get LS Ellipsoid estimate:	    
		self.LS_Ellps_fit_Coef =  ls_ellipsoid(self.X_orig_in.flatten(), self.Y_orig_in.flatten(), self.Z_orig_in.flatten() ) 
		self.LS_Ellps_center, self.LS_Ellps_axes, self.LS_Ellps_Rot_Mat, self.LS_Ellps_Inv_Rot_Mat = polyToParams3D(self.LS_Ellps_fit_Coef, False)   #get ellipsoid 3D parameters
	
		# Put our point cloud in LS Ellipsoid coordinates:
		self.U_coors_pts_in = np.zeros_like(self.mean_curvs_input)
		self.V_coors_pts_in = np.zeros_like(self.mean_curvs_input)

		a0 = self.LS_Ellps_axes[0] #semi-axis lengths
		a1 = self.LS_Ellps_axes[1]
		a2 = self.LS_Ellps_axes[2]

		self.U_coors_pts_in, self.V_coors_pts_in = Conv_3D_pts_to_Elliptical_Coors(a0, a1, a2, self.point_cloud_input, self.LS_Ellps_Inv_Rot_Mat, self.LS_Ellps_center)

		self.LS_Ellps_Mean_Curvs = Mean_Curvs_on_Ellipsoid(a0, a1, a2, self.U_coors_pts_in, self.V_coors_pts_in)

		# Compute Anis Stress/ Cell Stress from INPUT Curvatures:
		self.Anis_Stress_pts_UV_input = self.TwoGammaVal*self.mean_curvs_input
		H0_ellps_avg_ellps_UV_curvs = np.average(self.LS_Ellps_Mean_Curvs, axis=0) # 1st method of H0 computation, for Ellipsoid in UV points
		self.Anis_Cell_Stress_pts_UV_input = self.TwoGammaVal*(self.mean_curvs_input - self.LS_Ellps_Mean_Curvs - (self.H0_avg_input_curvs -  H0_ellps_avg_ellps_UV_curvs))

		# See LS ellipsoid in R^3 coordinates, compare to input points:
		self.X_LS_Ellps_pts, self.Y_LS_Ellps_pts, self.Z_LS_Ellps_pts = Conv_Elliptical_Coors_to_3D_pts(self.U_coors_pts_in, self.V_coors_pts_in, self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, self.LS_Ellps_center)

		# OUTWARD Normal vectors of Ellipsoid:
		self.LS_Ellps_normals = Ellipsoid_Level_Set_Normals(self.X_LS_Ellps_pts, self.Y_LS_Ellps_pts, self.Z_LS_Ellps_pts, self.LS_Ellps_fit_Coef)

		LS_Ellps_Err_X = self.X_LS_Ellps_pts - self.X_orig_in
		LS_Ellps_Err_Y = self.Y_LS_Ellps_pts - self.Y_orig_in
		LS_Ellps_Err_Z = self.Z_LS_Ellps_pts - self.Z_orig_in

		self.LS_Ellps_Err_vecs = np.hstack(( LS_Ellps_Err_X, LS_Ellps_Err_Y, LS_Ellps_Err_Z ))

		# We want to find largest Ellipsoid Axis to compare with final Droplet orientation:
		axis_1 = np.vstack(( Conv_Elliptical_Coors_to_3D_pts( np.array([[0.]]), np.array([[np.pi/2.]]), self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, np.zeros_like(self.LS_Ellps_center)) )) # x-axis
		len_axis_1 = np.linalg.norm(axis_1)

		axis_2 = np.vstack(( Conv_Elliptical_Coors_to_3D_pts( np.array([[np.pi/2.]]), np.array([[np.pi/2.]]), self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, np.zeros_like(self.LS_Ellps_center)) )) # y_axis
		len_axis_2 = np.linalg.norm(axis_2)

		axis_3 = np.vstack(( Conv_Elliptical_Coors_to_3D_pts( np.array([[0.]]), np.array([[0.]]), self.LS_Ellps_axes, self.LS_Ellps_Inv_Rot_Mat, np.zeros_like(self.LS_Ellps_center)) )) # z_axis
		len_axis_3 = np.linalg.norm(axis_3)

		self.Maj_Min_Axis = np.zeros((3)) # 1 for major axis, -1 for minor axis, for {X,Y,Z}

		if(len_axis_1 >= len_axis_2 and len_axis_1 >= len_axis_3):
			self.major_orientation_axis = axis_1
			self.Maj_Min_Axis[0] = 1.
			self.ellps_semi_axis_a = len_axis_1
			
			if(len_axis_2 <= len_axis_3): 
				self.Maj_Min_Axis[1] = -1.
				self.ellps_semi_axis_c = len_axis_2
				self.minor_orientation_axis = axis_2
				self.ellps_semi_axis_b = len_axis_3
				self.medial_orientation_axis = axis_3
			else:
				self.Maj_Min_Axis[2] = -1.
				self.ellps_semi_axis_c = len_axis_3
				self.minor_orientation_axis = axis_3
				self.ellps_semi_axis_b = len_axis_2
				self.medial_orientation_axis = axis_2

		elif(len_axis_2 >= len_axis_1 and len_axis_2 >= len_axis_3):
			self.major_orientation_axis = axis_2
			self.Maj_Min_Axis[1] = 1.
			self.ellps_semi_axis_a = len_axis_2

			if(len_axis_1 <= len_axis_3): 
				self.Maj_Min_Axis[0] = -1.
				self.ellps_semi_axis_c = len_axis_1
				self.minor_orientation_axis = axis_1
				self.ellps_semi_axis_b = len_axis_3
				self.medial_orientation_axis = axis_3
			else:
				self.Maj_Min_Axis[2] = -1.
				self.ellps_semi_axis_c = len_axis_3
				self.minor_orientation_axis = axis_3
				self.ellps_semi_axis_b = len_axis_1
				self.medial_orientation_axis = axis_1

		elif(len_axis_3 >= len_axis_2 and len_axis_3 >= len_axis_1):
			self.major_orientation_axis = axis_3
			self.Maj_Min_Axis[2] = 1.
			self.ellps_semi_axis_a = len_axis_3

			if(len_axis_2 <= len_axis_1): 
				self.Maj_Min_Axis[1] = -1.
				self.ellps_semi_axis_c = len_axis_2
				self.minor_orientation_axis = axis_2
				self.ellps_semi_axis_b = len_axis_1
				self.medial_orientation_axis = axis_1
			else:
				self.Maj_Min_Axis[0] = -1.
				self.ellps_semi_axis_c = len_axis_1
				self.minor_orientation_axis = axis_1
				self.ellps_semi_axis_b = len_axis_2
				self.medial_orientation_axis = axis_2

		else:
			print("\n"+"Error, major axis not discerned"+"\n")
			
		# Get matrix to change basis from cartesian to ellipsoidal coors:
		self.A_basis_mat = np.zeros((3, 3))
		self.A_basis_mat[0,0] = self.major_orientation_axis[0, 0] /self.ellps_semi_axis_a
		self.A_basis_mat[0,1] = self.major_orientation_axis[1, 0] /self.ellps_semi_axis_a
		self.A_basis_mat[0,2] = self.major_orientation_axis[2, 0] /self.ellps_semi_axis_a
		self.A_basis_mat[1,0] = self.medial_orientation_axis[0, 0] /self.ellps_semi_axis_b
		self.A_basis_mat[1,1] = self.medial_orientation_axis[1, 0] /self.ellps_semi_axis_b
		self.A_basis_mat[1,2] = self.medial_orientation_axis[2, 0] /self.ellps_semi_axis_b
		self.A_basis_mat[2,0] = self.minor_orientation_axis[0, 0] /self.ellps_semi_axis_c
		self.A_basis_mat[2,1] = self.minor_orientation_axis[1, 0] /self.ellps_semi_axis_c
		self.A_basis_mat[2,2] = self.minor_orientation_axis[2, 0] /self.ellps_semi_axis_c
		
		self.H_ellps_e_1 = self.ellps_semi_axis_a/(2.*self.ellps_semi_axis_b**2) +  self.ellps_semi_axis_a/(2.*self.ellps_semi_axis_c**2)
		self.H_ellps_e_2 = self.ellps_semi_axis_b/(2.*self.ellps_semi_axis_a**2) +  self.ellps_semi_axis_b/(2.*self.ellps_semi_axis_c**2)
		self.H_ellps_e_3 = self.ellps_semi_axis_c/(2.*self.ellps_semi_axis_a**2) +  self.ellps_semi_axis_c/(2.*self.ellps_semi_axis_b**2)

		#!!!! ASSUMES WE USE PYCOMPADRE, could redo with signed dists to center !!!!#
		#self.LS_Ellps_Err_signed = -1.* np.sum( np.multiply(self.PyCompadre_Normals_used, self.LS_Ellps_Err_vecs), axis = 1 )
		self.LS_Ellps_Err_signed = -1.*np.sum( np.multiply(self.LS_Ellps_normals, self.LS_Ellps_Err_vecs), axis = 1 )
		
		
		# Least-Squares Fit to Harmonics in {X,Y,Z}, using ellipsoidal coordinates:
		self.deg_fit_lbdv = Droplet_Input_Dict['deg_lbdv_fit'] # degree of harmonics we use:

		if(Droplet_Input_Dict['MAX_lbdv_fit_PTS'] == True): # If we use 5810 points, ... 
			self.num_lbdv_pts_fit = 5810
		else: # ... or hyper-interpolation quad pts
			self.num_lbdv_pts_fit = lbdv_i.look_up_lbdv_pts(self.deg_fit_lbdv+1)

		LBDV_Fit = lbdv_i.lbdv_info(self.deg_fit_lbdv, self.num_lbdv_pts_fit)

		# get info we need from Ellipsoid data fit in lbdv:
		self.H0_Ellpsoid, self.Gauss_Bonnet_Rel_Err_Ellps, self.H_Ellps_Manny_lbdv_pts, self.X_lbdv_ellps_pts, self.Y_lbdv_ellps_pts, self.Z_lbdv_ellps_pts = self.Ellipsoid_LBDV(LBDV_Fit)
		
		# use H0_Ellpsoid to calculate tissue stress projections:
		self.sigma_11_e = self.TwoGammaVal*(self.H_ellps_e_1 - self.H0_Ellpsoid)
		self.sigma_22_e = self.TwoGammaVal*(self.H_ellps_e_2 - self.H0_Ellpsoid)
		self.sigma_33_e = self.TwoGammaVal*(self.H_ellps_e_3 - self.H0_Ellpsoid)
		
		'''
		print("self.H_ellps_e_1 = "+str(self.H_ellps_e_1))
		print("self.H_ellps_e_2 = "+str(self.H_ellps_e_2))
		print("self.H_ellps_e_3 = "+str(self.H_ellps_e_3))
		print("self.H0_Ellpsoid = "+str(self.H0_Ellpsoid))
		print("self.sigma_11_e = "+str(self.sigma_11_e))
		print("self.sigma_22_e = "+str(self.sigma_22_e))
		print("self.sigma_33_e = "+str(self.sigma_33_e))
		'''
		
		# tissue stress tensor, elliptical coors:
		self.Tissue_Stress_Tens_Ellp_Coors = np.zeros((3,3))
		self.Tissue_Stress_Tens_Ellp_Coors[0,0] = self.sigma_11_e
		self.Tissue_Stress_Tens_Ellp_Coors[1,1] = self.sigma_22_e
		self.Tissue_Stress_Tens_Ellp_Coors[2,2] = self.sigma_33_e
		tr_sigma_ellps = self.sigma_11_e + self.sigma_22_e + self.sigma_33_e
		
		# cartesian tissue stress tensor:
		self.Tissue_Stress_Tens_Cart_Coors = np.dot( np.dot(self.A_basis_mat.T ,self.Tissue_Stress_Tens_Ellp_Coors), self.A_basis_mat)
		self.sigma_11_tissue_x =  self.Tissue_Stress_Tens_Cart_Coors[0,0]
		self.sigma_22_tissue_y =  self.Tissue_Stress_Tens_Cart_Coors[1,1]
		self.sigma_33_tissue_z =  self.Tissue_Stress_Tens_Cart_Coors[2,2]
		
		'''
		print("self.Tissue_Stress_Tens_Cart_Coors = "+str(self.Tissue_Stress_Tens_Cart_Coors))
		tr_sigma_cart = self.sigma_11_tissue_x + self.sigma_22_tissue_y + self.sigma_33_tissue_z
		print("tr_sigma_ellps = "+str(tr_sigma_ellps)+", tr_sigma_cart = "+str(tr_sigma_cart) )
		sys.exit()
		'''

		X_fit_sph_coef_vec, Y_fit_sph_coef_vec, Z_fit_sph_coef_vec = Least_Squares_Harmonic_Fit(self.deg_fit_lbdv, self.U_coors_pts_in, self.V_coors_pts_in, self.X_orig_in, self.Y_orig_in, self.Z_orig_in, False)

		X_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(X_fit_sph_coef_vec, self.deg_fit_lbdv)
		Y_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(Y_fit_sph_coef_vec, self.deg_fit_lbdv)
		Z_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(Z_fit_sph_coef_vec, self.deg_fit_lbdv)


		# Create SPH_func to represent X, Y, Z:
		X_fit_sph = sph_f.sph_func(X_fit_sph_coef_mat, self.deg_fit_lbdv)
		Y_fit_sph = sph_f.sph_func(Y_fit_sph_coef_mat, self.deg_fit_lbdv)
		Z_fit_sph = sph_f.sph_func(Z_fit_sph_coef_mat, self.deg_fit_lbdv)

		self.X_fit_sph_UV_pts = X_fit_sph.Eval_SPH(self.U_coors_pts_in, self.V_coors_pts_in)
		self.Y_fit_sph_UV_pts = Y_fit_sph.Eval_SPH(self.U_coors_pts_in, self.V_coors_pts_in)
		self.Z_fit_sph_UV_pts = Z_fit_sph.Eval_SPH(self.U_coors_pts_in, self.V_coors_pts_in)

		self.XYZ_sph_fit_UV_pts = np.hstack(( self.X_fit_sph_UV_pts, self.Y_fit_sph_UV_pts, self.Z_fit_sph_UV_pts ))

		# Plot SPH fit errors to point cloud:
		self.XYZ_fit_UV_err_vecs = np.hstack(( self.X_fit_sph_UV_pts - self.X_orig_in, self.Y_fit_sph_UV_pts - self.Y_orig_in, self.Z_fit_sph_UV_pts - self.Z_orig_in ))
		
		# Get {X,Y,Z} Coordinates at lebedev points, so we can leverage our code more efficiently (and uniformly) on surface:
		self.X_fit_lbdv_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(X_fit_sph, LBDV_Fit, 'A')
		self.Y_fit_lbdv_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(Y_fit_sph, LBDV_Fit, 'A')
		self.Z_fit_lbdv_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(Z_fit_sph, LBDV_Fit, 'A')

		# create manifold to calculate H, average H:
		Manny_Dict = {}
		Manny_Name_Dict = {} # sph point cloud at lbdv
		Manny_Name_Dict['X_lbdv_pts'] = self.X_fit_lbdv_pts
		Manny_Name_Dict['Y_lbdv_pts'] = self.Y_fit_lbdv_pts
		Manny_Name_Dict['Z_lbdv_pts'] = self.Z_fit_lbdv_pts

		Manny_Dict['Pickle_Manny_Data'] = False # BJG: until it works don't pickle
		Manny_Dict['Maniold_lbdv'] = LBDV_Fit

		Manny_Dict['Manifold_SPH_deg'] = self.deg_fit_lbdv
		Manny_Dict['use_manifold_name'] = False # we are NOT using named shapes in these tests
		Manny_Dict['Maniold_Name_Dict'] = Manny_Name_Dict # sph point cloud at lbdv

		Manny = mnfd.manifold(Manny_Dict)

		# Test orientation:
		self.Manny_lbdv_XYZ_pts = np.hstack(( self.X_fit_lbdv_pts, self.Y_fit_lbdv_pts, self.Z_fit_lbdv_pts ))
		centered_lbdv_XYZ_pts = self.Manny_lbdv_XYZ_pts - self.avg_position_in #!!!! BJG: MAY WANT TO RECALCULATE CENTER USING LBDV !!!!#

		self.Normal_X_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny.Normal_Vec_X_A_Pts, Manny.Normal_Vec_X_B_Pts, LBDV_Fit) 
		self.Normal_Y_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny.Normal_Vec_Y_A_Pts, Manny.Normal_Vec_Y_B_Pts, LBDV_Fit) 
		self.Normal_Z_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny.Normal_Vec_Z_A_Pts, Manny.Normal_Vec_Z_B_Pts, LBDV_Fit)

		self.Normals_XYZ_lbdv_pts = np.hstack(( self.Normal_X_lbdv_pts, self.Normal_Y_lbdv_pts, self.Normal_Z_lbdv_pts ))

		# Makre sure orientation is inward, so H is positive (for Ellipsoid, and small deviations):
		Orientations = np.sum( np.multiply(centered_lbdv_XYZ_pts,  self.Normals_XYZ_lbdv_pts), axis = 1)
		num_pos_orr = np.sum(Orientations.flatten() > 0)

		Orientation = 1. # unchanged (we want INWARD)
		if(num_pos_orr > .5*self.num_lbdv_pts_fit):
			Orientation = -1.

		# Use Gauss-Bonnet to test our resolution of the manifold:
		self.K_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny.K_A_pts, Manny.K_B_pts, LBDV_Fit)
		Gauss_Bonnet_Err = euc_kf.Integral_on_Manny(self.K_lbdv_pts, Manny, LBDV_Fit) - 4*np.pi
		self.Gauss_Bonnet_Rel_Err = abs(Gauss_Bonnet_Err)/(4*np.pi)
		
		self.good_frame = True #whether or not the frame is good 
		
		# cutoff for bad frame:
		if(self.Gauss_Bonnet_Rel_Err > 1.e-2): 
			self.good_frame = False
			return

		self.Mean_Curv_lbdv_pts = Orientation*euc_kf.Combine_Chart_Quad_Vals(Manny.H_A_pts, Manny.H_B_pts, LBDV_Fit)
	
		# 3rd way we estimate H0: arithmetic average of H at lbdv points: 
		self.H0_avg_lbdv_curvs = np.sum( self.Mean_Curv_lbdv_pts.flatten() )/ self.num_lbdv_pts_fit 

		# 4th way we estimate H0: Integratal of H (on surface) divided by surface area. This is the ONE WE USE in our analysis:
		Ones_pts = 1.*(np.ones_like(self.Mean_Curv_lbdv_pts))
		self.H0_Int_of_H_over_Area = euc_kf.Integral_on_Manny(self.Mean_Curv_lbdv_pts, Manny, LBDV_Fit) / euc_kf.Integral_on_Manny(Ones_pts, Manny, LBDV_Fit)

		# This is the 2\gamma*(H-H0 we USE FOR OUR anisotropic stress analysis:
		self.Anis_Stress_pts_lbdv = self.TwoGammaVal*(self.Mean_Curv_lbdv_pts - self.H0_Int_of_H_over_Area) 		

		# ellipsoid mean curvature, and deviation from surface mean curvature:
		self.H_ellps_lbdv_pts = Mean_Curvs_on_Ellipsoid(a0, a1, a2, LBDV_Fit.theta_pts, LBDV_Fit.phi_pts) 
		self.Anis_Cell_Stress_pts_lbdv = self.TwoGammaVal*(self.Mean_Curv_lbdv_pts - self.H_ellps_lbdv_pts - (self.H0_Int_of_H_over_Area- self.H0_Ellpsoid))
		self.Anis_Tissue_Stress_pts_lbdv = self.TwoGammaVal*(self.H_ellps_lbdv_pts -  self.H0_Ellpsoid)
 		#!!! SHOULD INCLUDE H_0 - H_0_ellps
		

		#### CALCULATE NORMALS, need manny for B-coors #####
		# take ders of basis, compute cross product, normalize
		X_fit_sph_dTheta =  X_fit_sph.Quick_Theta_Der()
		Y_fit_sph_dTheta =  Y_fit_sph.Quick_Theta_Der()
		Z_fit_sph_dTheta =  Z_fit_sph.Quick_Theta_Der()

		X_SPH_theta_UV_pts = X_fit_sph_dTheta.Eval_SPH(self.U_coors_pts_in, self.V_coors_pts_in)
		Y_SPH_theta_UV_pts = Y_fit_sph_dTheta.Eval_SPH(self.U_coors_pts_in, self.V_coors_pts_in)
		Z_SPH_theta_UV_pts = Z_fit_sph_dTheta.Eval_SPH(self.U_coors_pts_in, self.V_coors_pts_in)
	
		XYZ_theta_UV_pts = np.hstack(( X_SPH_theta_UV_pts, Y_SPH_theta_UV_pts, Z_SPH_theta_UV_pts )) 

		X_SPH_phi_UV_pts= X_fit_sph.Eval_SPH_Der_Phi(self.U_coors_pts_in, self.V_coors_pts_in)		
		Y_SPH_phi_UV_pts= Y_fit_sph.Eval_SPH_Der_Phi(self.U_coors_pts_in, self.V_coors_pts_in)
		Z_SPH_phi_UV_pts= Z_fit_sph.Eval_SPH_Der_Phi(self.U_coors_pts_in, self.V_coors_pts_in)		

		XYZ_phi_UV_pts = np.hstack(( X_SPH_phi_UV_pts, Y_SPH_phi_UV_pts, Z_SPH_phi_UV_pts )) 

		dTheta_cross_dPhi_UV_pts = np.cross(XYZ_theta_UV_pts, XYZ_phi_UV_pts)
		cross_prod_X, cross_prod_Y, cross_prod_Z = np.hsplit(dTheta_cross_dPhi_UV_pts, 3)

		Cross_Prod_Norms = np.sqrt(np.sum(np.multiply(dTheta_cross_dPhi_UV_pts, dTheta_cross_dPhi_UV_pts), axis = 1)).reshape(self.num_pts, 1)
		Chart_Used = np.where(Cross_Prod_Norms > 1.e-2, 1., -1.)

		# Use Manny to get Chart B info for Normals at UV pts:
		U_B_pts_eval, V_B_pts_eval = chrts.Coor_A_To_B(self.U_coors_pts_in, self.V_coors_pts_in) # rotate to get B-coors on sphere pullback

		X_SPH_theta_B_UV_pts = Manny.X_Bar_theta.Eval_SPH(U_B_pts_eval, V_B_pts_eval)
		X_SPH_phi_B_UV_pts =Manny.X_Bar.Eval_SPH_Der_Phi(U_B_pts_eval, V_B_pts_eval)
		Y_SPH_theta_B_UV_pts = Manny.Y_Bar_theta.Eval_SPH(U_B_pts_eval, V_B_pts_eval)
		Y_SPH_phi_B_UV_pts =Manny.Y_Bar.Eval_SPH_Der_Phi(U_B_pts_eval, V_B_pts_eval)
		Z_SPH_theta_B_UV_pts = Manny.Z_Bar_theta.Eval_SPH(U_B_pts_eval, V_B_pts_eval)
		Z_SPH_phi_B_UV_pts =Manny.Z_Bar.Eval_SPH_Der_Phi(U_B_pts_eval, V_B_pts_eval)

		XYZ_theta_B_UV_pts = np.hstack(( X_SPH_theta_B_UV_pts, Y_SPH_theta_B_UV_pts, Z_SPH_theta_B_UV_pts ))
		XYZ_phi_B_UV_pts = np.hstack(( X_SPH_phi_B_UV_pts, Y_SPH_phi_B_UV_pts, Z_SPH_phi_B_UV_pts ))

		dTheta_cross_dPhi_B_UV_pts = np.cross(XYZ_theta_B_UV_pts, XYZ_phi_B_UV_pts)
		cross_prod_X_B, cross_prod_Y_B, cross_prod_Z_B = np.hsplit(dTheta_cross_dPhi_B_UV_pts, 3)

		nor_dir_X = np.where(Cross_Prod_Norms > 1.e-2, cross_prod_X, cross_prod_X_B)
		nor_dir_Y = np.where(Cross_Prod_Norms > 1.e-2, cross_prod_Y, cross_prod_Y_B)
		nor_dir_Z = np.where(Cross_Prod_Norms > 1.e-2, cross_prod_Z, cross_prod_Z_B)

		normal_dirs_UV_pts = np.hstack(( nor_dir_X, nor_dir_Y, nor_dir_Z ))
		normal_dirs_UV_pts_norms = np.sqrt(np.sum(np.multiply(normal_dirs_UV_pts, normal_dirs_UV_pts), axis = 1)).reshape(self.num_pts, 1)
		self.Normal_Dirs_Fit_UV_pts = Orientation*np.divide(normal_dirs_UV_pts, normal_dirs_UV_pts_norms)

		#!!!! ASSUMES WE USE PYCOMPADRE, could redo with MANIFOLD NORMALS !!!!#
		#self.XYZ_fit_UV_signed_errs = -1.* np.sum( np.multiply(self.PyCompadre_Normals_used, self.XYZ_fit_UV_err_vecs), axis = 1 )
		self.XYZ_fit_UV_signed_errs = -1.* np.sum( np.multiply(self.Normal_Dirs_Fit_UV_pts, self.XYZ_fit_UV_err_vecs), axis = 1 )	
		'''
		self.XYZ_fit_UV_signed_errs_fit = -1.* np.sum( np.multiply(self.Normal_Dirs_Fit_UV_pts, self.XYZ_fit_UV_err_vecs), axis = 1 )	
		for pt in range(self.num_pts):
			print("pt ="+str(pt)+", Chrt = "+str(Chart_Used[pt,0])+"\n"+"    self.XYZ_fit_UV_signed_errs[pt] = "+str(self.XYZ_fit_UV_signed_errs[pt])+"\n"+"self.XYZ_fit_UV_signed_errs_fit[pt] = "+str( self.XYZ_fit_UV_signed_errs_fit[pt] )+"\n"+"% err = "+str( (self.XYZ_fit_UV_signed_errs_fit[pt] - self.XYZ_fit_UV_signed_errs[pt])*100/self.XYZ_fit_UV_signed_errs[pt] ))
		'''
		############################

		# Calculate 2\gamma*(H_3 - H_1), mirroring Ellipsoid max stress anisotropy:
		Avg_H_lbdv_X_axis = (self.Mean_Curv_lbdv_pts[0,0] + self.Mean_Curv_lbdv_pts[1,0])/2.	
		Avg_H_lbdv_Y_axis = (self.Mean_Curv_lbdv_pts[2,0] + self.Mean_Curv_lbdv_pts[3,0])/2.							
		Avg_H_lbdv_Z_axis = (self.Mean_Curv_lbdv_pts[4,0] + self.Mean_Curv_lbdv_pts[5,0])/2.
		
		self.Anis_Stress_Drop_Ellips_e1_e3_Axes = self.TwoGammaVal*np.sum(np.multiply(self.Maj_Min_Axis, np.array([ Avg_H_lbdv_X_axis, Avg_H_lbdv_Y_axis, Avg_H_lbdv_Z_axis ])))
		
		'''
		print("len_axis_1 = "+str(len_axis_1)+", len_axis_2 = "+str(len_axis_2)+", len_axis_3 = "+str(len_axis_3))
		print("self.Maj_Min_Axis = "+str(self.Maj_Min_Axis))
		'''
		
		# Need other drops for other principal axes projections:
		maj_axes_ind = np.where(self.Maj_Min_Axis == 1)[0][0]
		med_axes_ind = np.where(self.Maj_Min_Axis == 0)[0][0]
		min_axes_ind = np.where(self.Maj_Min_Axis == -1)[0][0]
		
		self.Maj_Med_Axis = np.zeros((3))
		self.Maj_Med_Axis[maj_axes_ind] = 1.
		self.Maj_Med_Axis[med_axes_ind] = -1.
		
		self.Med_Min_Axis = np.zeros((3))
		self.Med_Min_Axis[med_axes_ind] = 1.
		self.Med_Min_Axis[min_axes_ind] = -1.
		
		self.Anis_Stress_Drop_Ellips_e1_e2_Axes = self.TwoGammaVal*np.sum(np.multiply(self.Maj_Med_Axis, np.array([ Avg_H_lbdv_X_axis, Avg_H_lbdv_Y_axis, Avg_H_lbdv_Z_axis ])))
		self.Anis_Stress_Drop_Ellips_e2_e3_Axes = self.TwoGammaVal*np.sum(np.multiply(self.Med_Min_Axis, np.array([ Avg_H_lbdv_X_axis, Avg_H_lbdv_Y_axis, Avg_H_lbdv_Z_axis ])))
		
		'''
		print("self.Anis_Stress_Drop_Ellips_e1_e3_Axes = "+str(self.Anis_Stress_Drop_Ellips_e1_e3_Axes))
		print("self.Anis_Stress_Drop_Ellips_e1_e2_Axes = "+str(self.Anis_Stress_Drop_Ellips_e1_e2_Axes))
		print("self.Anis_Stress_Drop_Ellips_e2_e3_Axes = "+str(self.Anis_Stress_Drop_Ellips_e2_e3_Axes))
		
		print("self.Maj_Min_Axis = "+str(self.Maj_Min_Axis)+", maj_axes_ind = "+str(maj_axes_ind)+", med_axes_ind = "+str(med_axes_ind)+", min_axes_ind = "+str(min_axes_ind))
		print("self.Maj_Med_Axis = "+str(self.Maj_Med_Axis))
		print("self.Med_Min_Axis = "+str(self.Med_Min_Axis)) 
		sys.exit()
		'''

		# Use Radial formulation of Manifold (ORIGINAL pts) to estimate 3D Volume:
		self.deg_fit_vol = Droplet_Input_Dict['deg_lbdv_Vol_Int'] # degree of harmonics we use for volume estimate
		LBDV_VOL = LBDV_Fit #default if we use same degree basis for both
		self.num_lbdv_pts_Vol_Int = self.num_lbdv_pts_fit # with same number of pts
 
		if(self.deg_fit_vol != self.deg_fit_lbdv):
			self.num_lbdv_pts_Vol_Int = lbdv_i.look_up_lbdv_pts(self.deg_fit_vol+1)
		
			if(Droplet_Input_Dict['MAX_lbdv_vol_PTS'] == True): # If we use 5810 points, ... 
				self.num_lbdv_pts_Vol_Int = 5810

			LBDV_VOL = lbdv_i.lbdv_info(self.deg_fit_vol, self.num_lbdv_pts_Vol_Int)

		R_input_pts = np.sqrt(self.X_adj_in**2 + self.Y_adj_in**2 + self.Z_adj_in**2)

		Theta_Rad_pts = np.zeros(( self.num_pts, 1 ))
		Phi_Rad_pts = np.zeros(( self.num_pts, 1 ))

		for pt in range(self.num_pts):
			theta_pt, phi_pt = chrts.Cart_To_Coor_A(self.point_cloud_adj[pt, 0], self.point_cloud_adj[pt, 1], self.point_cloud_adj[pt,2])
			Theta_Rad_pts[pt, 0] = theta_pt
			Phi_Rad_pts[pt, 0] = phi_pt
		
		R_sph_coef_vec = Least_Squares_Harmonic_Fit_SINGLE(self.deg_fit_vol, Theta_Rad_pts, Phi_Rad_pts , R_input_pts, False)
		R_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(R_sph_coef_vec, self.deg_fit_vol)

		# Create SPH_func to represent R:
		R_sph = sph_f.sph_func(R_sph_coef_mat, self.deg_fit_vol)

		# Vals at Quad Pts for our Integration in spherical coordinates:
		R_sph_quad_vals = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(R_sph, LBDV_VOL, 'A')
		self.Vol_Int_S2 = sph_f.S2_Integral(R_sph_quad_vals**3/3., LBDV_VOL)
		
		# 5th way we estimate H0, by getting an 'average' radius from the Volume, assuming it's spherical:
		self.H0_from_Vol_Int = ((4.*np.pi)/(3.*self.Vol_Int_S2))**(1./3.) # approx ~1/R, for V ~ (4/3)*pi*R^3

		###############################################################
		# Radial manifold for correlation #
		# create radial manifold to calculate dists for temporal autocoors in spherical coordinates:
		
		Manny_Rad_Dict = {}
		Manny_Rad_Name_Dict = {} # sph point cloud at lbdv
		Manny_Rad_Name_Dict['X_lbdv_pts'] = LBDV_VOL.X*R_sph_quad_vals #self.X_adj_in
		Manny_Rad_Name_Dict['Y_lbdv_pts'] = LBDV_VOL.Y*R_sph_quad_vals #self.Y_adj_in
		Manny_Rad_Name_Dict['Z_lbdv_pts'] = LBDV_VOL.Z*R_sph_quad_vals #self.Z_adj_in

		Manny_Rad_Dict['Pickle_Manny_Data'] = False # BJG: until it works don't pickle
		Manny_Rad_Dict['Maniold_lbdv'] = LBDV_VOL

		Manny_Rad_Dict['Manifold_SPH_deg'] = self.deg_fit_vol
		Manny_Rad_Dict['use_manifold_name'] = False # we are NOT using named shapes in these tests
		Manny_Rad_Dict['Maniold_Name_Dict'] = Manny_Name_Dict # sph point cloud at lbdv

		Manny_Rad = mnfd.manifold(Manny_Rad_Dict)

		# Use Gauss-Bonnet to test our resolution of the Radial manifold:
		Rad_K_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Manny_Rad.K_A_pts, Manny_Rad.K_B_pts, LBDV_VOL)
		Gauss_Bonnet_Err_Rad = euc_kf.Integral_on_Manny(Rad_K_lbdv_pts, Manny_Rad, LBDV_VOL) - 4*np.pi
		self.Gauss_Bonnet_Rel_Err_Rad = abs(Gauss_Bonnet_Err_Rad)/(4*np.pi)

		Mean_Curv_Rad_lbdv_pts_unsigned = euc_kf.Combine_Chart_Quad_Vals(Manny_Rad.H_A_pts, Manny_Rad.H_B_pts, LBDV_VOL)

		# We estimate H0 on Radial Manifold: Integratal of H_Rad (on surface) divided by Rad surface area:
		Ones_pts = 1.*(np.ones_like(Mean_Curv_Rad_lbdv_pts_unsigned))
		H0_Rad_Int_of_H_over_Area = euc_kf.Integral_on_Manny(Mean_Curv_Rad_lbdv_pts_unsigned, Manny_Rad, LBDV_VOL) / euc_kf.Integral_on_Manny(Ones_pts, Manny_Rad, LBDV_VOL)

		abs_H0_Rad_Int_of_H_over_Area = abs(H0_Rad_Int_of_H_over_Area) # should be positive
		Mean_Curv_Rad_lbdv_pts = Mean_Curv_Rad_lbdv_pts_unsigned*abs_H0_Rad_Int_of_H_over_Area/H0_Rad_Int_of_H_over_Area # flip sign, if needed		
		self.Anis_Stress_pts_Rad_lbdv = self.TwoGammaVal*(Mean_Curv_Rad_lbdv_pts - abs_H0_Rad_Int_of_H_over_Area)
		
		# Error in H approx between manifolds:
		self.Mean_Curv_diff_Mannys = 0.5*( euc_kf.Integral_on_Manny(Mean_Curv_Rad_lbdv_pts - self.Mean_Curv_lbdv_pts, Manny_Rad, LBDV_VOL) + euc_kf.Integral_on_Manny(Mean_Curv_Rad_lbdv_pts - self.Mean_Curv_lbdv_pts, Manny, LBDV_Fit) )
		###############################################################
		

		# Compute Spatial Auto-Correlations in Mean Curvature, on lbdv data:
		self.dists_lbdv, self.tris_lbdv = plts.Triangulation_Inclusion_Distance_Max( self.Manny_lbdv_XYZ_pts, [], self.num_lbdv_pts_fit, MY_DIR)

		gdist_dists = sp_sprs.csr_matrix.toarray( gd.local_gdist_matrix( self.Manny_lbdv_XYZ_pts, np.int32(self.tris_lbdv)) ) # USE GDIST
		#print("gdist_dists.shape = "+str(gdist_dists.shape))
		#print("gdist_dists = "+str(gdist_dists))
		#print("self.dists_lbdv = "+str(self.dists_lbdv))
		self.dists_lbdv = gdist_dists # REPLACE WITH THIS

		self.corr_start_dist, self.auto_spat_corrs_microns_dists, self.mean_curvs_auto_corrs_avg, self.mean_curvs_auto_corr_norm, self.mean_curvs_auto_corrs_avg_normed = Corrs_From_Input_Tris_and_Dists(self.Anis_Stress_pts_lbdv, self.Anis_Stress_pts_lbdv, self.dists_lbdv, self.num_lbdv_pts_fit, Droplet_Input_Dict['Corr_Start_Dist'])

		# Compute Cellular Stress spatial correlations, (We don't need to save redundant fields from above):
		junk1, self.auto_cell_corrs_microns_dists, junk3, junk4, self.cell_stress_auto_corrs_avg_normed = Corrs_From_Input_Tris_and_Dists(self.Anis_Cell_Stress_pts_lbdv, self.Anis_Cell_Stress_pts_lbdv, self.dists_lbdv, self.num_lbdv_pts_fit, Droplet_Input_Dict['Corr_Start_Dist'])

		# Compute Tissue Stress spatial correlations, (We don't need to save redundant fields from above):
		junk1, self.auto_tiss_corrs_microns_dists, junk3, junk4, self.tissue_stress_auto_corrs_avg_normed = Corrs_From_Input_Tris_and_Dists(self.Anis_Tissue_Stress_pts_lbdv, self.Anis_Tissue_Stress_pts_lbdv, self.dists_lbdv, self.num_lbdv_pts_fit, Droplet_Input_Dict['Corr_Start_Dist'])

		#########################################################################
		# Do Local Max/Min analysis on 2\gamma*(H - H0) and 2\gamma*(H - H_ellps) data:
		self.local_max_and_min_AnisStress, self.nearest_min_max_dists_AnisStress, self.nearest_min_max_anisotropic_stress_AnisStress, self.ALL_min_max_dists_AnisStress, self.ALL_min_max_anisotropies_AnisStress = local_min_max_and_dists(self.Anis_Stress_pts_lbdv, self.dists_lbdv, self.tris_lbdv)
		self.local_max_and_min_AnisCellStress, self.nearest_min_max_dists_AnisCellStress, self.nearest_min_max_anisotropic_stress_AnisCellStress, self.ALL_min_max_dists_AnisCellStress, self.ALL_min_max_anisotropies_AnisCellStress = local_min_max_and_dists(self.Anis_Cell_Stress_pts_lbdv, self.dists_lbdv, self.tris_lbdv)

		# FIND CDF for 2\gamma*(H - H_0):
		self.min_val_excl_AnisStress, self.max_val_excl_AnisStress, self.curv_pts_excluded_AnisStress, self.hist_dist_Total_Stress_lbdv = CDF_Analysis_of_Data(self.Anis_Stress_pts_lbdv, Droplet_Input_Dict['alpha_percentile_excl_AnisStress'])

		self.num_curvs_above_max_AnisStress = np.count_nonzero(self.curv_pts_excluded_AnisStress == 1)
		self.num_curvs_below_min_AnisStress = np.count_nonzero(self.curv_pts_excluded_AnisStress == -1)

		# FIND CDF for 2\gamma*(H - H_ellps):
		self.min_val_excl_AnisCellStress, self.max_val_excl_AnisCellStress, self.curv_pts_excluded_AnisCellStress, self.hist_dist_Cell_Stress_lbdv = CDF_Analysis_of_Data(self.Anis_Cell_Stress_pts_lbdv, Droplet_Input_Dict['alpha_percentile_excl_AnisCellStress'])

		self.num_curvs_above_max_AnisCellStress = np.count_nonzero(self.curv_pts_excluded_AnisCellStress == 1)
		self.num_curvs_below_min_AnisCellStress = np.count_nonzero(self.curv_pts_excluded_AnisCellStress == -1)

		# Find CDF for 2\gamma*(H_Local_Max - H_Local_Min) distribution, using ALL pairs:
		self.min_val_excl_All_AnisStressLocalMax_m_AnisStressLocalMin, self.max_val_excl_All_AnisStressLocalMax_m_AnisStressLocalMin, self.curv_pts_excluded_All_AnisStressLocalMax_m_AnisStressLocalMin, self.hist_dist_All_AnisStressLocalMax_m_AnisStressLocalMin = CDF_Analysis_of_Data(self.ALL_min_max_anisotropies_AnisStress, Droplet_Input_Dict['alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin'])

		# FIND CDF for 2\gamma*(H_Input - H_0_Input):
		self.min_val_excl_AnisStress_Input_UV, self.max_val_excl_AnisStress_Input_UV, self.curv_pts_excluded_AnisStress_Input_UV, self.hist_dist_AnisStress_Input_UV = CDF_Analysis_of_Data(self.Anis_Stress_pts_UV_input, Droplet_Input_Dict['alpha_percentile_excl_AnisStress']) # same \alpha as corresponding lbdv field

		# FIND CDF for 2\gamma*(H_Input - H_ellps_{INPUT}):
		self.min_val_excl_AnisCellStress_Input_UV, self.max_val_excl_AnisCellStress_Input_UV, self.curv_pts_excluded_AnisCellStress_Input_UV, self.hist_dist_AnisCellStress_Input_UV = CDF_Analysis_of_Data(self.Anis_Cell_Stress_pts_UV_input, Droplet_Input_Dict['alpha_percentile_excl_AnisCellStress']) # same \alpha as corresponding lbdv field
		

		# look at which points exluded from 2\gamma*HmHe are also local min/max of 2\gamma*H:
		self.HmHe_exlc_vs_Local_H_min_max_pts = np.multiply( self.curv_pts_excluded_AnisCellStress.flatten(), self.local_max_and_min_AnisStress.flatten() )  # should be 1 where these align, 0 otherwise

		num_local_extrema_curvs_in_cutoff_AnisCellStress_align = np.count_nonzero(self.HmHe_exlc_vs_Local_H_min_max_pts == 1)
		num_local_extrema_curvs_in_cutoff_AnisCellStress_anti_align = np.count_nonzero(self.HmHe_exlc_vs_Local_H_min_max_pts == -1)

		# Analyze Signed Errors from Ellipsoid (!!!!NEED NORMAL pts on Ellipsoid!!!!):
		self.deg_fit_ESE = Droplet_Input_Dict['deg_fit_Ellipsoid_Deviation_Analysis'] # degree of harmonics we use:
		quad_fit_ESE = lbdv_i.look_up_lbdv_pts(self.deg_fit_ESE+1)
		LBDV_Fit_ESE = lbdv_i.lbdv_info(self.deg_fit_ESE, quad_fit_ESE)

		self.ESE_true_sph_coef_vec = Least_Squares_Harmonic_Fit_SINGLE(self.deg_fit_ESE, self.U_coors_pts_in, self.V_coors_pts_in, self.LS_Ellps_Err_signed, True)
		ESE_true_sph_coef_vec_abs = abs(self.ESE_true_sph_coef_vec)
		self.ESE_true_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(ESE_true_sph_coef_vec_abs, self.deg_fit_ESE) #!!!! NOT SPH fn, since we use actual harmonics here !!!!#
		#########################################################################


		# whether or not we plot vtu files for the analysis, which are called below:
		self.Plot_Vtu_Outputs = Droplet_Input_Dict['Plot_Vtu_Outputs']
		
		if(self.Plot_Vtu_Outputs == True):
			self.Plot_Input_and_PyCompadre_Data()
			self.Plot_Least_Squares_Ellipsoid()
			self.Plot_SPH_fit_UV()
			self.Plot_SPH_fit_lbdv()
			self.Plot_SPH_ELLPS_LBDV()
			
			if(self.Neha_Analysis):
				self.Plot_EK_Drop_Center()

	# Plot Drop center and EK as points:
	def Plot_EK_Drop_Center(self):
	
		drop_center = np.mean(self.Manny_lbdv_XYZ_pts, axis=0).reshape(3,1)
		EK_knot_coors = drop_center - self.Neha_Rad_Vec.T # get EK coors
		EK_DC_Coors = np.hstack(( drop_center, EK_knot_coors ))
		
		fieldData_list_EK_DC = [] # data we want to add

		fieldData_EK_DC = {};
		fieldData_EK_DC['fieldName'] = 'Is_EK';
		fieldData_EK_DC['fieldValues'] = [0, 1]
		fieldData_EK_DC['NumberOfComponents'] = 1;
		fieldData_list_EK_DC.append(fieldData_EK_DC);
		
		fieldData_EK_DC = {};
		fieldData_EK_DC['fieldName'] = 'Is_DC';
		fieldData_EK_DC['fieldValues'] = [1, 0]
		fieldData_EK_DC['NumberOfComponents'] = 1;
		fieldData_list_EK_DC.append(fieldData_EK_DC);		
		
		write_vtp_atz(self.EK_DC_Plot_file_path, EK_DC_Coors, fieldData_list_EK_DC) # write modified output of shape
		
	# First vtu plot: Read in Data, with PyCompadre info (if used):
	def Plot_Input_and_PyCompadre_Data(self):
		fieldData_list_plot1 = [] # data we want to add

		fieldData_plot1 = {};
		fieldData_plot1['fieldName'] = 'mean_curvs_in';
		fieldData_plot1['fieldValues'] = self.mean_curvs_input.flatten()
		fieldData_plot1['NumberOfComponents'] = 1;
		fieldData_list_plot1.append(fieldData_plot1);		
		
		fieldData_plot1 = {};
		fieldData_plot1['fieldName'] = 'U_coors_pts_in';
		fieldData_plot1['fieldValues'] = self.U_coors_pts_in.flatten()
		fieldData_plot1['NumberOfComponents'] = 1;
		fieldData_list_plot1.append(fieldData_plot1);
		
		fieldData_plot1 = {};
		fieldData_plot1['fieldName'] = 'V_coors_pts_in';
		fieldData_plot1['fieldValues'] = self.V_coors_pts_in.flatten()
		fieldData_plot1['NumberOfComponents'] = 1;
		fieldData_list_plot1.append(fieldData_plot1);


		if(self.Use_PyCompadre == True):

			fieldData_plot1 = {};
			fieldData_plot1['fieldName'] = 'PyComp_curvs_calc';
			fieldData_plot1['fieldValues'] = self.PyComp_Mean_Curvs.flatten()
			fieldData_plot1['NumberOfComponents'] = 1;
			fieldData_list_plot1.append(fieldData_plot1);

			fieldData_plot1 = {};
			fieldData_plot1['fieldName'] = 'Normals_Used';
			fieldData_plot1['fieldValues'] = self.PyCompadre_Normals_used.T
			fieldData_plot1['NumberOfComponents'] = 3;
			fieldData_list_plot1.append(fieldData_plot1);


		#print("self.Input_Plot_file_path = "+str(self.Input_Plot_file_path))
		write_vtp_atz(self.Input_Plot_file_path, self.point_cloud_input.T, fieldData_list_plot1) # write modified output of shape
	

	# Plot LS Ellipsoid, and relavant fields on it:
	def Plot_Least_Squares_Ellipsoid(self):
		fieldData_list_LS_Ellipsoid = [] # data we want to add

		fieldData_LSE = {};
		fieldData_LSE['fieldName'] = 'theta_pts';
		fieldData_LSE['fieldValues'] = self.U_coors_pts_in.flatten() #Theta_pts_used_test.flatten()
		fieldData_LSE['NumberOfComponents'] = 1;
		fieldData_list_LS_Ellipsoid.append(fieldData_LSE);

		fieldData_LSE = {};
		fieldData_LSE['fieldName'] = 'phi_pts';
		fieldData_LSE['fieldValues'] = self.V_coors_pts_in.flatten() #Phi_pts_used_test.flatten()
		fieldData_LSE['NumberOfComponents'] = 1;
		fieldData_list_LS_Ellipsoid.append(fieldData_LSE);

		fieldData_LSE = {};
		fieldData_LSE['fieldName'] = 'LS_Ellps_Err_signed';
		fieldData_LSE['fieldValues'] = self.LS_Ellps_Err_signed.flatten() 
		fieldData_LSE['NumberOfComponents'] = 1;
		fieldData_list_LS_Ellipsoid.append(fieldData_LSE);

		fieldData_LSE = {};
		fieldData_LSE['fieldName'] = 'LS_Ellps_Err_vecs';
		fieldData_LSE['fieldValues'] = self.LS_Ellps_Err_vecs.T
		fieldData_LSE['NumberOfComponents'] = 3;
		fieldData_list_LS_Ellipsoid.append(fieldData_LSE);
		
		
		# Plot modes of True SPH deviations INDIVIDUALLY:
		mode_num = 0
		for n in range(self.deg_fit_ESE+1):
			for m in range(-1*n, n+1):

				Y_mn_pts_ESE = sph_harm(m, n, self.U_coors_pts_in, self.V_coors_pts_in) #lbdv_i.Eval_SPH_Basis(m, n, U_pts_cloud, V_pts_cloud)
				Y_mn_Coef = self.ESE_true_sph_coef_vec[mode_num]
				
				Mode_mn_read_pts = (Y_mn_Coef*Y_mn_pts_ESE).real
				Mode_mn_label = "LS_Ellps_Err_Mode_Deg_"+str(n)+"_Ord_"+str(m).replace('-', "neg")+"_real"
				
				fieldData_LSE = {};
				fieldData_LSE['fieldName'] = Mode_mn_label;
				fieldData_LSE['fieldValues'] = Mode_mn_read_pts.flatten()
				fieldData_LSE['NumberOfComponents'] = 1;
				fieldData_list_LS_Ellipsoid.append(fieldData_LSE);

				mode_num = mode_num + 1
		

		fieldData_LSE = {};
		fieldData_LSE['fieldName'] = 'H_ellps_pts';
		fieldData_LSE['fieldValues'] = self.LS_Ellps_Mean_Curvs.flatten() 
		fieldData_LSE['NumberOfComponents'] = 1;
		fieldData_list_LS_Ellipsoid.append(fieldData_LSE);

		#print("self.Least_Square_Ellps_Plot_file_path = "+str(self.Least_Square_Ellps_Plot_file_path))
		write_vtp_atz(self.Least_Square_Ellps_Plot_file_path, np.hstack(( self.X_LS_Ellps_pts, self.Y_LS_Ellps_pts, self.Z_LS_Ellps_pts )).T , fieldData_list_LS_Ellipsoid) # write modified output of shape # LS_Ellipsoid_Pts.T


	# Plot SPH fit, with original (U,V) ellipsoidal coordinates for points:
	def Plot_SPH_fit_UV(self):
		# SPH plotted in orig coor:
		fieldData_list_LS_SPH = [] # data we want to add

		fieldData_LS_sph = {};
		fieldData_LS_sph['fieldName'] = 'XYZ_fit_UV_signed_errs';
		fieldData_LS_sph['fieldValues'] = self.XYZ_fit_UV_signed_errs.flatten() 
		fieldData_LS_sph['NumberOfComponents'] = 1;
		fieldData_list_LS_SPH.append(fieldData_LS_sph);

		fieldData_LS_sph = {};
		fieldData_LS_sph['fieldName'] = 'XYZ_fit_UV_err_vecs';
		fieldData_LS_sph['fieldValues'] = self.XYZ_fit_UV_err_vecs.T 
		fieldData_LS_sph['NumberOfComponents'] = 3;
		fieldData_list_LS_SPH.append(fieldData_LS_sph);

		#print("self.SPH_UV_fit_Plot_file_path = "+str(self.SPH_UV_fit_Plot_file_path))
		write_vtp_atz(self.SPH_UV_fit_Plot_file_path, self.XYZ_sph_fit_UV_pts.T, fieldData_list_LS_SPH)

	
	# Plot SPH fit, and lbdv points (in Ellipsoidal Coordinates), and fields we calculate here:
	def Plot_SPH_fit_lbdv(self):
		# LBDV_Fit plot:
		fieldData_list_LBDV_Fit = [] # data we want to add

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'H_ellps_pts_lbdv';
		fieldData_LBDV_Fit['fieldValues'] = self.H_ellps_lbdv_pts.flatten() 
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit); 

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'Anis_Cell_Stress_pts_lbdv'; 
		fieldData_LBDV_Fit['fieldValues'] = self.Anis_Cell_Stress_pts_lbdv.flatten() 
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);
		
		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'Anis_Tissue_Stress_pts_lbdv';
		fieldData_LBDV_Fit['fieldValues'] = self.Anis_Tissue_Stress_pts_lbdv.flatten() 
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'Mean_Curv_lbdv_pts';
		fieldData_LBDV_Fit['fieldValues'] = self.Mean_Curv_lbdv_pts.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'Anis_Stress_pts_lbdv';
		fieldData_LBDV_Fit['fieldValues'] = self.Anis_Stress_pts_lbdv
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'K_lbdv_pts';
		fieldData_LBDV_Fit['fieldValues'] = self.K_lbdv_pts.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'local_max_and_min_AnisStress';
		fieldData_LBDV_Fit['fieldValues'] = self.local_max_and_min_AnisStress.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'nearest_min_max_dists_AnisStress';
		fieldData_LBDV_Fit['fieldValues'] = self.nearest_min_max_dists_AnisStress.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);		

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'local_max_and_min_AnisCellStress';
		fieldData_LBDV_Fit['fieldValues'] = self.local_max_and_min_AnisCellStress.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);	
	
		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'nearest_min_max_dists_AnisCellStress';
		fieldData_LBDV_Fit['fieldValues'] = self.nearest_min_max_dists_AnisCellStress.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'curv_pts_excluded_AnisStress';
		fieldData_LBDV_Fit['fieldValues'] = self.curv_pts_excluded_AnisStress.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'curv_pts_excluded_AnisCellStress';
		fieldData_LBDV_Fit['fieldValues'] = self.curv_pts_excluded_AnisCellStress.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		fieldData_LBDV_Fit = {};
		fieldData_LBDV_Fit['fieldName'] = 'HmHe_exlc_vs_Local_H_min_max_pts';
		fieldData_LBDV_Fit['fieldValues'] = self.HmHe_exlc_vs_Local_H_min_max_pts.flatten()
		fieldData_LBDV_Fit['NumberOfComponents'] = 1;
		fieldData_list_LBDV_Fit.append(fieldData_LBDV_Fit);

		#print("plot_file_path_LBDV_Fit = "+str(self.SPH_lbdv_fit_Plot_file_path))
		write_vtp_atz(self.SPH_lbdv_fit_Plot_file_path, self.Manny_lbdv_XYZ_pts.T, fieldData_list_LBDV_Fit) # write modified output of shape


	# Plot LS Ellipsoid at LBDV points:
	def Plot_SPH_ELLPS_LBDV(self):
		fieldData_list_LBDV_ELLPS = [] # data we want to add

		fieldData_LBDV_ELLPS = {};
		fieldData_LBDV_ELLPS['fieldName'] = 'H_Ellps_Manny_lbdv_pts';
		fieldData_LBDV_ELLPS['fieldValues'] = self.H_Ellps_Manny_lbdv_pts.flatten() 
		fieldData_LBDV_ELLPS['NumberOfComponents'] = 1;
		fieldData_list_LBDV_ELLPS.append(fieldData_LBDV_ELLPS);

		fieldData_LBDV_ELLPS = {};
		fieldData_LBDV_ELLPS['fieldName'] = 'Anis_Tissue_Stress_pts_lbdv';
		fieldData_LBDV_ELLPS['fieldValues'] = self.Anis_Tissue_Stress_pts_lbdv.flatten() 
		fieldData_LBDV_ELLPS['NumberOfComponents'] = 1;
		fieldData_list_LBDV_ELLPS.append(fieldData_LBDV_ELLPS);

		write_vtp_atz(self.SPH_lbdv_ellps_fit_Plot_file_path, np.hstack(( self.X_lbdv_ellps_pts, self.Y_lbdv_ellps_pts, self.Z_lbdv_ellps_pts )).T, fieldData_list_LBDV_ELLPS)

	
	# Use same lbdv_fit to analyze ellipsoid on lbdv points:
	def Ellipsoid_LBDV(self, LBDV_Fit):

		X_ellps_fit_sph_coef_vec, Y_ellps_fit_sph_coef_vec, Z_ellps_fit_sph_coef_vec = Least_Squares_Harmonic_Fit(self.deg_fit_lbdv, self.U_coors_pts_in, self.V_coors_pts_in, self.X_LS_Ellps_pts, self.Y_LS_Ellps_pts, self.Z_LS_Ellps_pts, False)

		X_ellps_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(X_ellps_fit_sph_coef_vec, self.deg_fit_lbdv)
		Y_ellps_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(Y_ellps_fit_sph_coef_vec, self.deg_fit_lbdv)
		Z_ellps_fit_sph_coef_mat = sph_f.Un_Flatten_Coef_Vec(Z_ellps_fit_sph_coef_vec, self.deg_fit_lbdv)

		# Create SPH_func to represent X, Y, Z:
		X_ellps_fit_sph = sph_f.sph_func(X_ellps_fit_sph_coef_mat, self.deg_fit_lbdv)
		Y_ellps_fit_sph = sph_f.sph_func(Y_ellps_fit_sph_coef_mat, self.deg_fit_lbdv)
		Z_ellps_fit_sph = sph_f.sph_func(Z_ellps_fit_sph_coef_mat, self.deg_fit_lbdv)

		# Get {X,Y,Z} Ellps Coordinates at lebedev points, so we can leverage our code:
		X_lbdv_ellps_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(X_ellps_fit_sph, LBDV_Fit, 'A')
		Y_lbdv_ellps_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(Y_ellps_fit_sph, LBDV_Fit, 'A')
		Z_lbdv_ellps_pts = euc_kf.Extract_Quad_Pt_Vals_From_SPH_Fn(Z_ellps_fit_sph, LBDV_Fit, 'A')

		# create manifold of Ellps, to calculate H and average H of Ellps:
		Ellps_Manny_Dict = {}
		Ellps_Manny_Name_Dict = {} # sph point cloud at lbdv
		Ellps_Manny_Name_Dict['X_lbdv_pts'] = X_lbdv_ellps_pts
		Ellps_Manny_Name_Dict['Y_lbdv_pts'] = Y_lbdv_ellps_pts
		Ellps_Manny_Name_Dict['Z_lbdv_pts'] = Z_lbdv_ellps_pts

		Ellps_Manny_Dict['Pickle_Manny_Data'] = False # BJG: until it works don't pickle
		Ellps_Manny_Dict['Maniold_lbdv'] = LBDV_Fit

		Ellps_Manny_Dict['Manifold_SPH_deg'] = self.deg_fit_lbdv
		Ellps_Manny_Dict['use_manifold_name'] = False # we are NOT using named shapes in these tests
		Ellps_Manny_Dict['Maniold_Name_Dict'] = Ellps_Manny_Name_Dict # sph point cloud at lbdv

		Ellps_Manny = mnfd.manifold(Ellps_Manny_Dict)

		# Use Gauss-Bonnet to test our resolution of the manifold:
		Ellps_K_lbdv_pts = euc_kf.Combine_Chart_Quad_Vals(Ellps_Manny.K_A_pts, Ellps_Manny.K_B_pts, LBDV_Fit)
		Gauss_Bonnet_Err_Ellps = euc_kf.Integral_on_Manny(Ellps_K_lbdv_pts, Ellps_Manny, LBDV_Fit) - 4*np.pi
		Gauss_Bonnet_Rel_Err_Ellps = abs(Gauss_Bonnet_Err_Ellps)/(4*np.pi)

		Mean_Curv_Ellps_lbdv_pts_unsigned = euc_kf.Combine_Chart_Quad_Vals(Ellps_Manny.H_A_pts, Ellps_Manny.H_B_pts, LBDV_Fit)

		# We estimate H0 on Ellps: Integratal of H_ellps (on surface) divided by Ellps surface area:
		Ones_pts = 1.*(np.ones_like(Mean_Curv_Ellps_lbdv_pts_unsigned))
		H0_Ellps_Int_of_H_over_Area = euc_kf.Integral_on_Manny(Mean_Curv_Ellps_lbdv_pts_unsigned, Ellps_Manny, LBDV_Fit) / euc_kf.Integral_on_Manny(Ones_pts, Ellps_Manny, LBDV_Fit)

		abs_H0_Ellps_Int_of_H_over_Area = abs(H0_Ellps_Int_of_H_over_Area) # should be positive
		Mean_Curv_Ellps_lbdv_pts = Mean_Curv_Ellps_lbdv_pts_unsigned*abs_H0_Ellps_Int_of_H_over_Area/H0_Ellps_Int_of_H_over_Area # flip sign, if needed
		
		return abs_H0_Ellps_Int_of_H_over_Area, Gauss_Bonnet_Rel_Err_Ellps, Mean_Curv_Ellps_lbdv_pts, X_lbdv_ellps_pts, Y_lbdv_ellps_pts, Z_lbdv_ellps_pts


	# gets info stored in Droplet Class Dict:
	def get_Drop_Info(self):
		return self.Drop_Info_Dict
