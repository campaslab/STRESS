#! This will use Droplet_Analsis Objects, and iterate over these, to compute time-dependent statisics (!!!! SHOULD BE AN OBJECT TOO !!!!):

import os, sys
import numpy as np
import re as re # for getting droplet number label from filenames we read in
from scipy.io import savemat # for writing .mat file, for Elijah's code to read
from scipy.io import loadmat # for reading .mat file

import Refactored_Droplet_Class as Droplet_Analysis # use these objects for time-sequences:
import plots_SPB as plts


# make this something we can call from MATLAB:
#def Batch_Droplet_Class_Analysis(Drop_Anal_Input_Dict):

# Need to read in from matfile format first:
def Load_Dict_From_Matfile(input_matfile_filepath):

	from pathlib import Path

	input_matfile_loaded = loadmat(input_matfile_filepath)
	print("input_matfile_loaded = "+str(input_matfile_loaded))
	print("input_matfile_loaded['SubFolder_We_Load_From']= "+str(Path(str(input_matfile_loaded['SubFolder_We_Load_From'][0]))))
	print("input_matfile_loaded['MatFile_filepath']= "+str(Path(input_matfile_loaded['MatFile_filepath'][0])))
	Loaded_Dict = {}
	
	# Ones that have to be set in MATLAB:
	Loaded_Dict['SubFolder_We_Load_From'] = str(input_matfile_loaded['SubFolder_We_Load_From'][0])
	Loaded_Dict['MatFile_filepath'] = Path(str(input_matfile_loaded['MatFile_filepath'][0]))
	Loaded_Dict['Tension_gamma_val'] = float(input_matfile_loaded['Tension_gamma_val'][0,0])
	Loaded_Dict['Plot_Vtu_Outputs'] = bool(input_matfile_loaded['Plot_Vtu_Outputs'][0,0])
	Loaded_Dict['deg_lbdv_fit'] = int(input_matfile_loaded['deg_lbdv_fit'][0,0])
	Loaded_Dict['MAX_lbdv_fit_PTS'] = bool(input_matfile_loaded['MAX_lbdv_fit_PTS'][0,0])
	Loaded_Dict['alpha_percentile_excl_AnisStress'] = float(input_matfile_loaded['alpha_percentile_excl_AnisStress'][0,0])
	Loaded_Dict['Neha_Radial_analysis'] = bool(input_matfile_loaded['Neha_Radial_analysis'][0,0])
	Loaded_Dict['Pixel_Size_Microns'] = float(input_matfile_loaded['Pixel_Size_Microns'][0,0])
	
	# set from read inputs:
	Loaded_Dict['Output_Plot_Subfolder_Name'] = Loaded_Dict['SubFolder_We_Load_From']+"_OUTPUT" #'Test_of_Refactor_Plots' # directory for plotting outputs
	Loaded_Dict['deg_lbdv_Vol_Int'] = int(Loaded_Dict['deg_lbdv_fit']) #11
	Loaded_Dict['MAX_lbdv_vol_PTS'] = bool(Loaded_Dict['MAX_lbdv_fit_PTS']) #False
	Loaded_Dict['deg_fit_Ellipsoid_Deviation_Analysis']  = int(Loaded_Dict['deg_lbdv_fit']) # degree for analyzing modes of deviation from Ellipsoid
	Loaded_Dict['alpha_percentile_excl_AnisCellStress'] = float(Loaded_Dict['alpha_percentile_excl_AnisStress'])
	Loaded_Dict['alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin'] = float(Loaded_Dict['alpha_percentile_excl_AnisStress'])
	
	# set for all inputs
	Loaded_Dict['Use_PyCompadre'] = False
	Loaded_Dict['PyCompadre_p_order'] = int(2)
	Loaded_Dict['Corr_Start_Dist'] = int(0) #int(3) # distance at which we start looking at correlations (since there is a noisy spike at dist = 0)
	Loaded_Dict['Calc_Spat_Temp_Corrs'] = False #bool(input_matfile_loaded['Calc_Spat_Temp_Corrs'][0,0])
	Loaded_Dict['smoothBSpline'] = True #bool(input_matfile_loaded['smoothBSpline'][0,0])

	#return Loaded_Dict
	Batch_Droplet_Class_Analysis(Loaded_Dict)


# Create class for SINGLE analyzed Droplet Point Cloud:
class Batch_Droplet_Class_Analysis(object):

	def __init__(self, Drop_Anal_Input_Dict):

		#!!! TEST THIS CODE !!!!#
		directory = Drop_Anal_Input_Dict['MatFile_filepath'] #os.fsencode(directory_in_str)

		Dict_of_All_Drops = {} # we save each to a dictionary, using the number as a key
		Output_Dir = [] # get from first drop

		Last_major_axis_orientation = [] # get semi-major axis of last drop:
		H0_from_V_Drop_0 = [] # use inverse for sphere radius for temporal auto-correlations

		# for each drop, save LS Ellisoid shape for .mat file:
		LS_Ellps_axes_vecs = []
		LS_Ellps_Inv_Rot_mats = []
		LS_Ellps_center_vecs = []
		
		first_good_index = [] # index of first good frame:
		indicies_used = [] #list of indicies used
		num_pts_frames = [] # list of num seg pts

		# find maximum degree fit we can use:
		for index, file in enumerate(sorted(os.listdir(directory))):
			filename = os.fsdecode(file)
			if filename.endswith(".mat"): 
			
				# Load specified .mat file, read in point cloud and Elijah's calculation of mean curvature:
				mat_file_path = os.path.join(directory, filename) #os.path.join(MAT_Files_Dir, mat_file_name)

				mat_contents = loadmat(mat_file_path)
				num_pts_ind = []
				
				# If we have input .mat file of type 'XYZ_H_and_pixelSize'
				if 'data' in mat_contents.keys(): # 'px_sz_um' 
					num_pts_frames.append( len(mat_contents['data']['XYZ_um'][0, 0]) )
				
				# If we have input .mat file of type 'justCoordinatesAndCurvatures00...'
				else:
					num_pts_frames.append( len(mat_contents['pointCloudArray']) )
			
		min_num_seg_pts = min(num_pts_frames) # we are limited by worst segmentation
		max_degree_allowed = int(np.sqrt(min_num_seg_pts) - 1.0)
		
		print("num_pts_frames = "+str(num_pts_frames)+", min_num_seg_pts = "+str(min_num_seg_pts)+", max_degree_allowed = "+str(max_degree_allowed))
		
		max_degree_usable = []
		max_degree_usable_name = ""
		
		# figure out maximum resolution
		if(max_degree_allowed >= 20):
			max_degree_usable = 20
			max_degree_usable_name = "VERY HIGH"
		elif(max_degree_allowed >= 17):
			max_degree_usable = 17
			max_degree_usable_name = "HIGH"
		elif(max_degree_allowed >= 14):
			max_degree_usable = 14
			max_degree_usable_name = "MEDIUM"
		elif(max_degree_allowed >= 11):
			max_degree_usable = 11
			max_degree_usable_name = "LOW"
		elif(max_degree_allowed >= 8):
			max_degree_usable = 8
			max_degree_usable_name = "SUPER LOW"
		elif(max_degree_allowed >= 5):
			max_degree_usable = 5
			max_degree_usable_name = "BOTTOM OF THE BARREL"
		else:
			print("\n"+"ERROR: SEGMENTATION HAS TO FEW POINTS TO ANALYZE"+"\n")
			return
			
		# if we need to lower basis degree:
		if(Drop_Anal_Input_Dict['deg_lbdv_fit'] > max_degree_usable):
			print("\n"+"WARNING: TOO FEW SEGMENTED POINTS TO USE DEGREE "+str(Drop_Anal_Input_Dict['deg_lbdv_fit'])+" BASIS, "+"\n"+"LOWERING TO DEGREE "+str(max_degree_usable)+" ("+str(max_degree_usable_name)+") RESOLUTION"+"\n")
			Drop_Anal_Input_Dict['deg_lbdv_fit'] = max_degree_usable
			Drop_Anal_Input_Dict['deg_lbdv_Vol_Int'] = max_degree_usable
			Drop_Anal_Input_Dict['deg_fit_Ellipsoid_Deviation_Analysis'] = max_degree_usable

	
		for index, file in enumerate(sorted(os.listdir(directory))):
			filename = os.fsdecode(file)
			if filename.endswith(".mat"): 
				#print("INCLUDING (filename): "+ str(filename))

				file_num = (re.findall('\d+', filename ))[0]

				Drop_Anal_Input_Dict['MatFile_name'] = filename
				Drop_i = Droplet_Analysis.Droplet_Analysis(Drop_Anal_Input_Dict)
				
				if(Drop_i.good_frame == True):
				
					print("INCLUDING: file_num = "+str(file_num)+", index = "+str(index))	
					indicies_used.append(index)
					
					# see if this is the first good frame:
					if(first_good_index == []): 
						first_good_index = index
				
					Dict_of_All_Drops[str(file_num)] = Drop_i

					Last_major_axis_orientation = Drop_i.major_orientation_axis # gives us last value

					# Get LS_Ellipsoid info:
					LS_Ellps_axes_vecs.append(Drop_i.LS_Ellps_axes)
					LS_Ellps_Inv_Rot_mats.append(Drop_i.LS_Ellps_Inv_Rot_Mat)
					LS_Ellps_center_vecs.append(Drop_i.LS_Ellps_center)

					print("Gauss-Bonnet: "+str(filename)+", = "+str( Drop_i.Gauss_Bonnet_Rel_Err ))	

					if(index == first_good_index): #0): #!!!! CHANGE TO FIRST DROP USABLE !!!!#
						Output_Dir = Drop_i.Output_Plot_Subfolder_path
						H0_from_V_Drop_0 = Drop_i.H0_from_Vol_Int
						#print("\n"+"Output_Dir = "+str(Output_Dir)+"\n")
						Drop_Anal_Input_Dict['Pixel_Size_Microns'] = Drop_i.pixel_size_microns

						## DO PLOTS on Single Drop Data ##:
						
						# only look at upper triangular part, since this is symetric:
						Upper_Tri_ESE_true_sph_coef_mat = np.triu(Drop_i.ESE_true_sph_coef_mat + np.triu(Drop_i.ESE_true_sph_coef_mat, k=1))
						Normalized_UpperTr_Coefs = plts.Plot_Matrix_Heatmap_Linear_Adj("Drop 0 Upper Triangular Ellipsoidal Deviation Contributions", Upper_Tri_ESE_true_sph_coef_mat, 0., Only_Upper_Tri=True, Input_Dir=Output_Dir)
						
						# flip for plotting:
						Flipped_UT_mat = np.zeros_like(Upper_Tri_ESE_true_sph_coef_mat)
						for col_j in range(Drop_i.deg_fit_ESE + 1):
							for row_i in range(col_j+1):
								Flipped_UT_mat[row_i, col_j] = Upper_Tri_ESE_true_sph_coef_mat[col_j-row_i, col_j]

						plts.Plot_Matrix_Heatmap_Linear_Adj("FLIPPED Drop 0 Upper Triangular Ellipsoidal Deviation Contributions", Flipped_UT_mat, 0., Only_Upper_Tri=True, Input_Dir=Output_Dir)

						Normalized_UpperTr_Coefs = Normalized_UpperTr_Coefs[np.triu_indices(Drop_i.deg_fit_ESE + 1)]
						sorted_mode_contributions = np.sort( Normalized_UpperTr_Coefs.flatten() )[::-1]
						plts.Plot_Histogram_CDF_ONLY("Drop 0 Ellipsoidal Deviation Mode Contributions", np.arange( ((Drop_i.deg_fit_ESE+1)*(Drop_i.deg_fit_ESE + 2))/2).flatten() +1, sorted_mode_contributions, Output_Dir)

						# Histrograms on stress distributions:
						plts.Plot_Histogram_PDF("Total Stress extrema pair distances ", Drop_i.nearest_min_max_dists_AnisStress[np.nonzero(Drop_i.nearest_min_max_dists_AnisStress)].flatten(), Output_Dir )
						plts.Plot_Histogram_PDF("Total Stress extrema pair vals ", Drop_i.nearest_min_max_anisotropic_stress_AnisStress[np.nonzero(Drop_i.nearest_min_max_anisotropic_stress_AnisStress)].flatten(), Output_Dir )

						plts.Plot_Histogram_PDF("Total Stress ALL pair distances ", Drop_i.ALL_min_max_dists_AnisStress.flatten(), Output_Dir )
						plts.Plot_Histogram_PDF("Total Stress extrema ALL pair vals ", Drop_i.ALL_min_max_anisotropies_AnisStress.flatten(), Output_Dir )

						plts.Plot_Histogram_PDF("Cell Stress extrema pair distances ", Drop_i.nearest_min_max_dists_AnisCellStress[np.nonzero(Drop_i.nearest_min_max_dists_AnisCellStress)].flatten(), Output_Dir )
						plts.Plot_Histogram_PDF("Cell Stress extrema pair vals ", Drop_i.nearest_min_max_anisotropic_stress_AnisCellStress[np.nonzero(Drop_i.nearest_min_max_anisotropic_stress_AnisCellStress)].flatten(), Output_Dir )

						plts.Plot_Histogram_PDF("Cell Stress extrema ALL pair distances ", Drop_i.ALL_min_max_dists_AnisCellStress.flatten(), Output_Dir )
						plts.Plot_Histogram_PDF("Cell Stress extrema ALL pair vals ", Drop_i.ALL_min_max_anisotropies_AnisCellStress.flatten(), Output_Dir )

						# scatter plot of all pairs, with anisoptopies against distance:
						plts.SimpleScatterPlot("Total Stress All local min max pair anisotropies vs pair distances", Drop_i.ALL_min_max_dists_AnisStress.flatten(),  Drop_i.ALL_min_max_anisotropies_AnisStress.flatten(), "distances", "anisotropies", plot_type = 'x', plot_dir=Output_Dir)
						plts.SimpleScatterPlot("Cell Stress local min max pair anisotropies vs pair distances", Drop_i.ALL_min_max_dists_AnisCellStress.flatten(), Drop_i.ALL_min_max_anisotropies_AnisCellStress.flatten(), "distances", "anisotropies", plot_type = 'x', plot_dir=Output_Dir)

						# sorted stress distibution:
						sorted_total_stress = np.sort(Drop_i.Anis_Stress_pts_lbdv.flatten()) 
						plts.plot_scalar_fields_over_time("total stress distribution", np.arange(Drop_i.num_lbdv_pts_fit).reshape(Drop_i.num_lbdv_pts_fit, 1), sorted_total_stress.reshape(Drop_i.num_lbdv_pts_fit, 1), ["Total Stress"], Input_Dir=Output_Dir)

						sorted_cell_stress = np.sort(Drop_i.Anis_Cell_Stress_pts_lbdv.flatten()) 
						plts.plot_scalar_fields_over_time("cell stress distribution", np.arange(Drop_i.num_lbdv_pts_fit).reshape(Drop_i.num_lbdv_pts_fit, 1), sorted_cell_stress.reshape(Drop_i.num_lbdv_pts_fit, 1), ["Cell Stress"], Input_Dir=Output_Dir)

						X_prob_TotalStress = np.linspace(sorted_total_stress[0], sorted_total_stress[-1], Drop_i.num_lbdv_pts_fit)
						plts.Plot_Histograms("Total Stress Cutoff CDF", X_prob_TotalStress, Drop_i.hist_dist_Total_Stress_lbdv, sorted_total_stress, Drop_Anal_Input_Dict['alpha_percentile_excl_AnisStress'], Drop_i.min_val_excl_AnisStress, Drop_i.max_val_excl_AnisStress, Output_Dir)

						X_prob_CellStress = np.linspace(sorted_cell_stress[0], sorted_cell_stress[-1], Drop_i.num_lbdv_pts_fit)
						plts.Plot_Histograms("Cell Stress Cutoff CDF", X_prob_CellStress, Drop_i.hist_dist_Cell_Stress_lbdv, sorted_cell_stress, Drop_Anal_Input_Dict['alpha_percentile_excl_AnisCellStress'], Drop_i.min_val_excl_AnisStress, Drop_i.max_val_excl_AnisCellStress, Output_Dir)
						
				else:
					print("NOT USING: file_num = "+str(file_num)+", index = "+str(index)+", Gauss-Bonnet = "+str(Drop_i.Gauss_Bonnet_Rel_Err))	

			else:
				print("\n"+"NOT INCLUDING: "+ str(os.path.join(directory, filename))+", which is in specfied folder"+"\n")
				continue

		Num_Drops = len(Dict_of_All_Drops)
		print("Num_Drops = "+str(Num_Drops))
		array_of_indicies_used = np.array(indicies_used, dtype=np.dtype('d')).reshape(Num_Drops, 1)
		print("array_of_indicies_used = "+str(array_of_indicies_used)+"\n")
		
		# if there is NO good data:
		if(Num_Drops == 0):
			print("\n"+"ERROR: NO FRAMES ABLE TO BE ANALYZED (GO BACK TO THE LAB AND GET BETTER DATA)"+"\n")
			return

		# smoothing wont work without enough droplets in the sequence:
		if(Num_Drops < 5):
			print("NOT ENOUGH DROPS TO SMOOTH PLOTS")
			Drop_Anal_Input_Dict['smoothBSpline'] = False

		#print("\n"+"DROPLET DICT NAMES: "+str(Dict_of_All_Drops.keys())+"\n")

		Num_Dropets_Seg_Pts_Over_Time = []
		Gauss_Bonnet_Test_Over_Time = []
		Volumes_Measured_Over_Time = []
		Anis_Stress_Alpha_Percentile_Dif_Over_Time = []
		AnisCellStress_Alpha_Percentile_Dif_Over_Time = []
		Anis_Stress_Ellps_Max_Difs_Over_Time = []
		Anis_Stress_Drop_Ellps_Axes_Over_Time = []
		Anis_Stress_Drop_Ellps_e1_e2_Over_Time = []
		Anis_Stress_Drop_Ellps_e2_e3_Over_Time = []
		All_AnisStressLocalMax_m_AnisStressLocalMin_Alpha_Percentile_Dif_Over_Time = []
		Abs_Cos_Orient_Over_Time = []
		Neha_abs_cos_btw_rad_e1_Over_Time = []
		Neha_Drop_rad_Dist_EK = []
		Neha_DropCent_to_EK_vecs = []
		abs_cos_e_1_x_hat_Over_Time = []
		abs_cos_e_1_y_hat_Over_Time = []
		abs_cos_e_1_z_hat_Over_Time = []
		Tissue_Stress_x_proj_Over_Time = []
		Tissue_Stress_y_proj_Over_Time = []
		Tissue_Stress_z_proj_Over_Time = []
		H0_lbdv_surf_minus_Ellps_Over_Time = []
		Anis_Stress_Input_Alpha_Percentile_Dif_Over_Time = []
		AnisCellStress_Input_Alpha_Percentile_Dif_Over_Time = []

		Spatial_Corrs_dists_list = []
		Cell_Corrs_dists_list = []
		Tiss_Corrs_dists_list = []
		Spatial_Corrs_corrs_list = []
		Spatial_Corrs_labels_list = []
		Spatial_Normed_Corrs_corrs_list = []

		Gauss_Bonnet_Rad_Test_Over_Time = []
		H_lbdv_vs_H_Rad_Over_Time = []

		Anis_Stress_Rad_pts = []
		Cell_Stress_Rad_pts = []
		Tissue_Stress_Rad_pts = []

		Spatial_Normed_Cell_Corrs_corrs_list = []
		Spatial_Normed_Tissue_Corrs_corrs_list = []

		# self.H0_from_Vol_Int
		# self.H0_Int_of_H_over_Area 
		#self.H0_Ellpsoid 
		#self.H0_avg_lbdv_curv
		# self.H0_avg_input_curvs
		H0_From_Vol_Int = []
		H0_From_Area_Int = []
		H0_From_Ellipsoid = []
		H0_From_avg_lbdv_pts = []
		H0_From_avg_seg_pts = []

		Haversine_S2_lbdv_vol_dists = [] # need matrix of distances for spatio-temp corrs on S2:
		S2_adj_rad = 1./H0_from_V_Drop_0 #30. # to make this big enough to use integer methods


		# Iterate over data, for analyzing temporal data:
		for index_i, key_i in enumerate(sorted(Dict_of_All_Drops.keys())):
			#print("index_i ="+str(index_i)+", key_i = "+str(key_i))
			Drop_i = Dict_of_All_Drops[key_i]

			if(index_i == first_good_index): #0):
				Haversine_S2_lbdv_vol_dists = S2_adj_rad*Droplet_Analysis.Haversine_dists_S2_lbdv(Drop_i.deg_fit_vol, Drop_i.num_lbdv_pts_Vol_Int)

			num_seg_pts_i = Drop_i.num_pts
			Num_Dropets_Seg_Pts_Over_Time.append(num_seg_pts_i)

			Vol_i = Drop_i.Vol_Int_S2
			#print("Vol_i = "+str(Vol_i))
			Volumes_Measured_Over_Time.append(Vol_i)

			Anis_Stress_alpha_percentile_range_i = Drop_i.max_val_excl_AnisStress - Drop_i.min_val_excl_AnisStress
			Anis_Stress_Alpha_Percentile_Dif_Over_Time.append(Anis_Stress_alpha_percentile_range_i)

			AnisCellStress_percentile_range_i = Drop_i.max_val_excl_AnisCellStress - Drop_i.min_val_excl_AnisCellStress
			AnisCellStress_Alpha_Percentile_Dif_Over_Time.append(AnisCellStress_percentile_range_i)

			All_AnisStressLocalMax_m_AnisStressLocalMin_percentile_range_i = Drop_i.max_val_excl_All_AnisStressLocalMax_m_AnisStressLocalMin - Drop_i.min_val_excl_All_AnisStressLocalMax_m_AnisStressLocalMin
			All_AnisStressLocalMax_m_AnisStressLocalMin_Alpha_Percentile_Dif_Over_Time.append(All_AnisStressLocalMax_m_AnisStressLocalMin_percentile_range_i)

			semi_axis_sorted = np.sort(Drop_i.LS_Ellps_axes)
			a = semi_axis_sorted[2]
			b = semi_axis_sorted[1]
			c = semi_axis_sorted[0]

			Theoretical_Anis_Stress_Ellps_Max_Dif = Drop_Anal_Input_Dict['Tension_gamma_val']*(a/(c**2) + (a-c)/(b**2) - c/(a**2))
			#print("H_ellps_Max_Difs_range_i = "+str(H_ellps_Max_Difs_range_i)+", Theoretical_Anis_Stress_Ellps_Max_Dif = "+str(Theoretical_Anis_Stress_Ellps_Max_Dif))

			Anis_Stress_Ellps_Max_Dif_range_i = Theoretical_Anis_Stress_Ellps_Max_Dif #max(Drop_i.LS_Ellps_Mean_Curvs) - min(Drop_i.LS_Ellps_Mean_Curvs)
			Anis_Stress_Ellps_Max_Difs_Over_Time.append(Anis_Stress_Ellps_Max_Dif_range_i)

			Anis_Stress_Drop_Ellps_Extrm_Axis_i = Drop_i.Anis_Stress_Drop_Ellips_e1_e3_Axes
			Anis_Stress_Drop_Ellps_Axes_Over_Time.append(Anis_Stress_Drop_Ellps_Extrm_Axis_i)
			Anis_Stress_Drop_Ellps_e1_e2_Over_Time.append(Drop_i.Anis_Stress_Drop_Ellips_e1_e2_Axes)
			Anis_Stress_Drop_Ellps_e2_e3_Over_Time.append(Drop_i.Anis_Stress_Drop_Ellips_e2_e3_Axes)

			# Check orientation with final orientation:
			abs_cos_orient_i = abs(np.sum(np.multiply(Last_major_axis_orientation, Drop_i.major_orientation_axis)))/(np.linalg.norm(Last_major_axis_orientation, 2)*np.linalg.norm(Drop_i.major_orientation_axis, 2))
			Abs_Cos_Orient_Over_Time.append(abs_cos_orient_i)
			
			if(Drop_Anal_Input_Dict['Neha_Radial_analysis'] == True):
				Neha_cos_btw_rad_e1_i = np.sum(np.multiply(Drop_i.Neha_Rad_Vec.flatten(), Drop_i.major_orientation_axis.flatten()))/(np.linalg.norm(Drop_i.Neha_Rad_Vec.flatten(), 2)*np.linalg.norm(Drop_i.major_orientation_axis.flatten(), 2))
				#print("Drop_i.Neha_Rad_Vec = "+str(Drop_i.Neha_Rad_Vec)+", Drop_i.major_orientation_axis = "+str(Drop_i.major_orientation_axis))
				Neha_abs_cos_btw_rad_e1_Over_Time.append(abs(Neha_cos_btw_rad_e1_i))
				#print("index_i ="+str(index_i)+", key_i = "+str(key_i)+", Neha_cos_btw_rad_e1_i = "+str(Neha_cos_btw_rad_e1_i)+", Neha_abs_cos_btw_rad_e1_Over_Time = "+str(Neha_abs_cos_btw_rad_e1_Over_Time))
				
				Neha_Drop_rad_Dist_EK.append(Drop_i.len_Neha_Rad_Vec)
				Neha_DropCent_to_EK_vecs.append( Drop_i.Neha_Rad_Vec )

			# Look at e_1 orientation with x, y, z axes:
			abs_e_1_x_over_a_i = abs(Drop_i.major_orientation_axis.flatten()[0]/Drop_i.ellps_semi_axis_a)
			abs_e_1_y_over_a_i = abs(Drop_i.major_orientation_axis.flatten()[1]/Drop_i.ellps_semi_axis_a)
			abs_e_1_z_over_a_i = abs(Drop_i.major_orientation_axis.flatten()[2]/Drop_i.ellps_semi_axis_a)

			abs_cos_e_1_x_hat_Over_Time.append(abs_e_1_x_over_a_i)
			abs_cos_e_1_y_hat_Over_Time.append(abs_e_1_y_over_a_i)
			abs_cos_e_1_z_hat_Over_Time.append(abs_e_1_z_over_a_i)
			
			Tissue_Stress_x_proj_Over_Time.append(Drop_i.sigma_11_tissue_x)
			Tissue_Stress_y_proj_Over_Time.append(Drop_i.sigma_22_tissue_y)
			Tissue_Stress_z_proj_Over_Time.append(Drop_i.sigma_33_tissue_z)

			# gauss bonnet:
			Gauss_Bonnet_test_i = Drop_i.Gauss_Bonnet_Rel_Err
			Gauss_Bonnet_Test_Over_Time.append(Gauss_Bonnet_test_i)

			H0_minus_H0_Ellps_i = (Drop_i.H0_Int_of_H_over_Area- Drop_i.H0_Ellpsoid)
			H0_lbdv_surf_minus_Ellps_Over_Time.append(H0_minus_H0_Ellps_i)

			# Analysis on input (Elijah) Curvatures:
			Anis_Stress_Input_alpha_percentile_range_i = Drop_i.max_val_excl_AnisStress_Input_UV - Drop_i.min_val_excl_AnisStress_Input_UV
			Anis_Stress_Input_Alpha_Percentile_Dif_Over_Time.append(Anis_Stress_Input_alpha_percentile_range_i)

			AnisCellStress_Input_percentile_range_i = Drop_i.max_val_excl_AnisCellStress_Input_UV - Drop_i.min_val_excl_AnisCellStress_Input_UV
			AnisCellStress_Input_Alpha_Percentile_Dif_Over_Time.append(AnisCellStress_Input_percentile_range_i)

			Spatial_Corrs_dists_list.append(Drop_i.auto_spat_corrs_microns_dists)
			Cell_Corrs_dists_list.append(Drop_i.auto_cell_corrs_microns_dists)
			Tiss_Corrs_dists_list.append(Drop_i.auto_tiss_corrs_microns_dists)

			Spatial_Corrs_corrs_list.append(Drop_i.mean_curvs_auto_corrs_avg)
			Spatial_Corrs_labels_list.append("Drop_"+str(index_i))

			# Normed spatial corrs and Cell corrs:
			Spatial_Normed_Corrs_corrs_list.append(Drop_i.mean_curvs_auto_corrs_avg_normed)
			Spatial_Normed_Cell_Corrs_corrs_list.append(Drop_i.cell_stress_auto_corrs_avg_normed)
			Spatial_Normed_Tissue_Corrs_corrs_list.append(Drop_i.tissue_stress_auto_corrs_avg_normed)

			# Test Radial Manifold Accuracy:
			Gauss_Bonnet_Rad_Test_Over_Time.append(Drop_i.Gauss_Bonnet_Rel_Err_Rad)

			# Test diffs in H between manifold in Ellipsoid and Radial params
			H_lbdv_vs_H_Rad_Over_Time.append(Drop_i.Mean_Curv_diff_Mannys)

			# Anis Stesses and Cell-Stresses Measured at Radial Manny pts, for temporal correlations:
			Anis_Stress_Rad_pts.append(Drop_i.Anis_Stress_pts_Rad_lbdv.flatten())
			Cell_Stress_Rad_pts.append(Drop_i.Anis_Cell_Stress_pts_lbdv.flatten())
			Tissue_Stress_Rad_pts.append(Drop_i.Anis_Tissue_Stress_pts_lbdv.flatten())

			# record H0 for analysis:
			H0_From_Vol_Int.append(Drop_i.H0_from_Vol_Int)
			H0_From_Area_Int.append(Drop_i.H0_Int_of_H_over_Area )
			H0_From_Ellipsoid.append(Drop_i.H0_Ellpsoid)
			H0_From_avg_lbdv_pts.append(Drop_i.H0_avg_lbdv_curvs)
			H0_From_avg_seg_pts.append(Drop_i.H0_avg_input_curvs)

		# Calculate Temporal Corrs of total stresses:
		Temp_Inner_Prod_Anis_Stress_Rad = np.zeros(( Num_Drops, Num_Drops )) # _{ij} = H(x,t_{j-i})^T*H(x, t_{i}), j >= i for temporal corrs
		Temp_Inner_Prod_Cell_Stress_Rad = np.zeros(( Num_Drops, Num_Drops )) # same as above, but for cellular stresses
		Temp_Inner_Prod_Tissue_Stress_Rad = np.zeros(( Num_Drops, Num_Drops )) # same as above, but for tissue stresses

		for i_temp_corrs in range(Num_Drops):
			for j_temp_corrs in range(i_temp_corrs, Num_Drops):
				Temp_Inner_Prod_Anis_Stress_Rad[i_temp_corrs, j_temp_corrs] = np.sum(Anis_Stress_Rad_pts[j_temp_corrs - i_temp_corrs]*Anis_Stress_Rad_pts[j_temp_corrs]	)
				Temp_Inner_Prod_Cell_Stress_Rad[i_temp_corrs, j_temp_corrs] = np.sum(Cell_Stress_Rad_pts[j_temp_corrs - i_temp_corrs]*Cell_Stress_Rad_pts[j_temp_corrs]	)
				Temp_Inner_Prod_Tissue_Stress_Rad[i_temp_corrs, j_temp_corrs] = np.sum(Tissue_Stress_Rad_pts[j_temp_corrs - i_temp_corrs]*Tissue_Stress_Rad_pts[j_temp_corrs]	)

		summed_Temporal_Corrs_AnisStress_Rad = np.sum(Temp_Inner_Prod_Anis_Stress_Rad, axis=1) # sum vals of cols for each row
		summed_Temporal_Corrs_Cell_Stress_Rad = np.sum(Temp_Inner_Prod_Cell_Stress_Rad, axis=1)
		summed_Temporal_Corrs_Tissue_Stress_Rad = np.sum(Temp_Inner_Prod_Tissue_Stress_Rad, axis=1)

		Temporal_Corrs_AnisStress_Rad = [] # corrs for \tau
		Temporal_Corrs_Cell_Stress_Rad = []
		Temporal_Corrs_Tissue_Stress_Rad = []

		for tau in range(Num_Drops):
			Temporal_Corrs_AnisStress_Rad.append( ( summed_Temporal_Corrs_AnisStress_Rad[tau]/(Num_Drops - tau) )/( summed_Temporal_Corrs_AnisStress_Rad[0]/(Num_Drops) ) )
			Temporal_Corrs_Cell_Stress_Rad.append( ( summed_Temporal_Corrs_Cell_Stress_Rad[tau]/(Num_Drops - tau) )/( summed_Temporal_Corrs_Cell_Stress_Rad[0]/(Num_Drops) ) )
			Temporal_Corrs_Tissue_Stress_Rad.append( ( summed_Temporal_Corrs_Tissue_Stress_Rad[tau]/(Num_Drops - tau) )/( summed_Temporal_Corrs_Tissue_Stress_Rad[0]/(Num_Drops) ) )

		# Calculate Spatio-Temporal Corrs of total stresses:
		if(Drop_Anal_Input_Dict['Calc_Spat_Temp_Corrs']):
			S2_diam = int(np.pi*S2_adj_rad)
			junk1, junk2, junk3, junk4, junk5 = Droplet_Analysis.Corrs_From_Input_Tris_and_Dists(Anis_Stress_Rad_pts[0], Anis_Stress_Rad_pts[0], Haversine_S2_lbdv_vol_dists, len(Anis_Stress_Rad_pts[0]), Drop_Anal_Input_Dict['Corr_Start_Dist'] )
			num_dist_vals = len(junk2.flatten())
			SpatTemp_Inner_Prod_Anis_Stress_Rad = np.zeros(( Num_Drops, Num_Drops, num_dist_vals ))

			for i_temp_corrs in range(Num_Drops):
				for j_temp_corrs in range(i_temp_corrs, Num_Drops):

					junk1, junk2, Corr_jmi_j_unnormed, junk4, Corr_jmi_j_normed = Droplet_Analysis.Corrs_From_Input_Tris_and_Dists(Anis_Stress_Rad_pts[j_temp_corrs - i_temp_corrs], Anis_Stress_Rad_pts[j_temp_corrs], Haversine_S2_lbdv_vol_dists, len(Anis_Stress_Rad_pts[j_temp_corrs]), Drop_Anal_Input_Dict['Corr_Start_Dist'] )
					
					print("spat temp mat: ("+str(i_temp_corrs)+", "+str(j_temp_corrs)+") = corr("+str(j_temp_corrs - i_temp_corrs)+", "+str(j_temp_corrs)+")")

					SpatTemp_Inner_Prod_Anis_Stress_Rad[i_temp_corrs, j_temp_corrs, :] = Corr_jmi_j_unnormed.flatten()

			summed_SpatTemp_Corrs_AnisStress_Rad = np.squeeze(np.sum(SpatTemp_Inner_Prod_Anis_Stress_Rad, axis=1)) # sum vals of cols for each row, gives us a matrix
			print("summed_SpatTemp_Corrs_AnisStress_Rad.shape = "+str(summed_SpatTemp_Corrs_AnisStress_Rad.shape))
			num_tau_samples = (np.arange(Num_Drops)+1)[::-1].reshape(Num_Drops, 1)
			print("num_tau_samples = "+str(num_tau_samples))
			avg_summed_SpatTemp_Corrs_AnisStress_Rad = np.divide(summed_SpatTemp_Corrs_AnisStress_Rad, num_tau_samples)
			norm_t_0 = np.sum( summed_SpatTemp_Corrs_AnisStress_Rad[0, :].flatten() )
			print("norm_t_0 = "+str(norm_t_0))
			normed_avg_summed_SpatTemp_Corrs_AnisStress_Rad = avg_summed_SpatTemp_Corrs_AnisStress_Rad/norm_t_0

			print("summed_SpatTemp_Corrs_AnisStress_Rad = "+str(summed_SpatTemp_Corrs_AnisStress_Rad))
			print("avg_summed_SpatTemp_Corrs_AnisStress_Rad = "+str(avg_summed_SpatTemp_Corrs_AnisStress_Rad))
			print("normed_avg_summed_SpatTemp_Corrs_AnisStress_Rad = "+str(normed_avg_summed_SpatTemp_Corrs_AnisStress_Rad))

			Row_vals = np.array([np.arange(Num_Drops),]*num_dist_vals).T
			Col_vals = np.array([np.arange(num_dist_vals) + Drop_Anal_Input_Dict['Corr_Start_Dist'],]*Num_Drops)
			print("Row_vals = "+str(Row_vals))
			print("Col_vals = "+str(Col_vals))
			plts.Plot3DWireFrame("Spatio-Temp Corrs ", Row_vals, Col_vals, normed_avg_summed_SpatTemp_Corrs_AnisStress_Rad, ["Tau", "dist", "Spatio-Temps_Corrs"], plot_dir = Output_Dir)
			#plts.Plot_3D_Heatmap("Spatio-Temp Corrs ", Row_vals, Col_vals, normed_avg_summed_SpatTemp_Corrs_AnisStress_Rad, ["Tau", "dist", "Spatio-Temps_Corrs"], plot_dir = Output_Dir)
			#sys.exit()


		# Recast as arrays, for plotting:
		Num_Dropets_Seg_Pts_Over_Time = np.array(Num_Dropets_Seg_Pts_Over_Time, dtype=np.dtype('d'))
		Volumes_Measured_Over_Time = np.array(Volumes_Measured_Over_Time, dtype=np.dtype('d'))
		Anis_Stress_Alpha_Percentile_Dif_Over_Time = np.array(Anis_Stress_Alpha_Percentile_Dif_Over_Time, dtype=np.dtype('d'))
		AnisCellStress_Alpha_Percentile_Dif_Over_Time = np.array(AnisCellStress_Alpha_Percentile_Dif_Over_Time, dtype=np.dtype('d'))
		Anis_Stress_Ellps_Max_Difs_Over_Time = np.array(Anis_Stress_Ellps_Max_Difs_Over_Time, dtype=np.dtype('d'))
		Anis_Stress_Drop_Ellps_Axes_Over_Time = np.array(Anis_Stress_Drop_Ellps_Axes_Over_Time, dtype=np.dtype('d'))
		Anis_Stress_Drop_Ellps_e1_e2_Over_Time = np.array(Anis_Stress_Drop_Ellps_e1_e2_Over_Time, dtype=np.dtype('d'))
		Anis_Stress_Drop_Ellps_e2_e3_Over_Time = np.array(Anis_Stress_Drop_Ellps_e2_e3_Over_Time, dtype=np.dtype('d'))
		All_AnisStressLocalMax_m_AnisStressLocalMin_Alpha_Percentile_Dif_Over_Time = np.array(All_AnisStressLocalMax_m_AnisStressLocalMin_Alpha_Percentile_Dif_Over_Time, dtype=np.dtype('d'))
		Abs_Cos_Orient_Over_Time = np.array(Abs_Cos_Orient_Over_Time, dtype=np.dtype('d'))
		abs_cos_e_1_x_hat_Over_Time = np.array(abs_cos_e_1_x_hat_Over_Time, dtype=np.dtype('d'))
		abs_cos_e_1_y_hat_Over_Time = np.array(abs_cos_e_1_y_hat_Over_Time, dtype=np.dtype('d'))
		abs_cos_e_1_z_hat_Over_Time = np.array(abs_cos_e_1_z_hat_Over_Time, dtype=np.dtype('d'))

		Tissue_Stress_x_proj_Over_Time = np.array(Tissue_Stress_x_proj_Over_Time, dtype=np.dtype('d'))
		Tissue_Stress_y_proj_Over_Time = np.array(Tissue_Stress_y_proj_Over_Time, dtype=np.dtype('d'))
		Tissue_Stress_z_proj_Over_Time = np.array(Tissue_Stress_z_proj_Over_Time, dtype=np.dtype('d'))

		if(Drop_Anal_Input_Dict['Neha_Radial_analysis'] == True):
			Neha_abs_cos_btw_rad_e1_Over_Time = np.array(Neha_abs_cos_btw_rad_e1_Over_Time, dtype=np.dtype('d'))
			Neha_Drop_rad_Dist_EK = np.array(Neha_Drop_rad_Dist_EK, dtype=np.dtype('d'))
			Neha_DropCent_to_EK_vecs = np.array(Neha_DropCent_to_EK_vecs, dtype=np.dtype('d'))
		Gauss_Bonnet_Test_Over_Time = np.array(Gauss_Bonnet_Test_Over_Time, dtype=np.dtype('d'))
		H0_lbdv_surf_minus_Ellps_Over_Time = np.array(H0_lbdv_surf_minus_Ellps_Over_Time, dtype=np.dtype('d'))
		Anis_Stress_Input_Alpha_Percentile_Dif_Over_Time = np.array(Anis_Stress_Input_Alpha_Percentile_Dif_Over_Time, dtype=np.dtype('d'))
		AnisCellStress_Input_Alpha_Percentile_Dif_Over_Time = np.array(AnisCellStress_Input_Alpha_Percentile_Dif_Over_Time, dtype=np.dtype('d'))
		Gauss_Bonnet_Rad_Test_Over_Time = np.array(Gauss_Bonnet_Rad_Test_Over_Time, dtype=np.dtype('d'))
		H_lbdv_vs_H_Rad_Over_Time = np.array(H_lbdv_vs_H_Rad_Over_Time, dtype=np.dtype('d'))
		Temporal_Corrs_AnisStress_Rad = np.array(Temporal_Corrs_AnisStress_Rad, dtype=np.dtype('d'))
		Temporal_Corrs_Cell_Stress_Rad = np.array(Temporal_Corrs_Cell_Stress_Rad, dtype=np.dtype('d'))
		Temporal_Corrs_Tissue_Stress_Rad = np.array(Temporal_Corrs_Tissue_Stress_Rad, dtype=np.dtype('d'))

		H0_From_Vol_Int = np.array(H0_From_Vol_Int, dtype=np.dtype('d'))
		H0_From_Area_Int = np.array(H0_From_Area_Int, dtype=np.dtype('d'))
		H0_From_Ellipsoid = np.array(H0_From_Ellipsoid, dtype=np.dtype('d'))
		H0_From_avg_lbdv_pts = np.array(H0_From_avg_lbdv_pts, dtype=np.dtype('d'))
		H0_From_avg_seg_pts = np.array(H0_From_avg_seg_pts, dtype=np.dtype('d'))


		# plot num pts input from segmented data:
		plts.plot_scalar_fields_over_time(" Seg Pts Input Over Time", array_of_indicies_used, Num_Dropets_Seg_Pts_Over_Time.reshape(Num_Drops, 1), ["Num Seg Pts"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir)

		# plot volumes:
		plts.plot_scalar_fields_over_time(" Volume Over Time", array_of_indicies_used, Volumes_Measured_Over_Time.reshape(Num_Drops, 1), ["Drop Volumes"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir)

		# plot volumes with y_min = 0:
		plts.plot_scalar_fields_over_time(" Volume Over Time (From V=0)", array_of_indicies_used, Volumes_Measured_Over_Time.reshape(Num_Drops, 1), ["Drop Volumes"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, y_lim_bottom_in=0.)

		# plot 2\gamma*(H - H0) Alpha percentile ranges:
		plts.plot_scalar_fields_over_time(" Anis Stress Percentile Range Over Time", array_of_indicies_used, Anis_Stress_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), ["alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisStress'])], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot 2\gamma*(H-H_ellps) Alpha percentile ranges:
		plts.plot_scalar_fields_over_time(" Anis Cell Stresses Alpha Percentile Range Over Time", array_of_indicies_used, AnisCellStress_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), ["alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisCellStress'])], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot 2\gamma*(H_ellps_max - H_ellps_min) ranges:
		plts.plot_scalar_fields_over_time(" Maximum Anisotropy in Ellipsoidal Stress Over Time", array_of_indicies_used, Anis_Stress_Ellps_Max_Difs_Over_Time.reshape(Num_Drops, 1), ["max. Ellps. Stress Ans."], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot 2\gamma*(H(e_1) - H(e_3)) over time:
		plts.plot_scalar_fields_over_time(" Anisotropy in Drop Stress in Ellipsoid Axes Extrema Over Time", array_of_indicies_used, Anis_Stress_Drop_Ellps_Axes_Over_Time.reshape(Num_Drops, 1), ["2\gamma (H[e_1] - H[e_3])"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		hstacked_anis_stress_ellps_axes = np.hstack(( Anis_Stress_Drop_Ellps_Axes_Over_Time.reshape(Num_Drops, 1), Anis_Stress_Drop_Ellps_e1_e2_Over_Time.reshape(Num_Drops, 1), Anis_Stress_Drop_Ellps_e2_e3_Over_Time.reshape(Num_Drops, 1) ))

		max_anis_stress_ellps_axis_dir = max(hstacked_anis_stress_ellps_axes.flatten())
		min_anis_stress_ellps_axis_dir = min(hstacked_anis_stress_ellps_axes.flatten())

		# Plot a BUNCH of total anis stresses between ellipsoid axes dirs:
		plts.plot_scalar_fields_over_time("Anis (Total) Stress along Ellps Axes Dirs Over Time", array_of_indicies_used, hstacked_anis_stress_ellps_axes, ["2\gamma (H[e_1] - H[e_3])", "2\gamma (H[e_1] - H[e_2])", "2\gamma (H[e_2] - H[e_3])"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'], y_lim_bottom_in=float(min_anis_stress_ellps_axis_dir), y_lim_top_in=float(max_anis_stress_ellps_axis_dir) )


		# plot 2\gamma*(H_local_max-H_local_min) Alpha percentile ranges, for all pairs:
		plts.plot_scalar_fields_over_time("All pairs: (Local Anis Stress Max minus Local Anis Stress Min) Alpha Percentile Range Over Time", array_of_indicies_used, All_AnisStressLocalMax_m_AnisStressLocalMin_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), ["alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin'])], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot orientation wrt final axis:
		plts.plot_scalar_fields_over_time(" Abs(cos) Major Axis Dot Prod. with Final", array_of_indicies_used, Abs_Cos_Orient_Over_Time.reshape(Num_Drops, 1), ["abs(cos) angle"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot orientation wrt x axis:
		plts.plot_scalar_fields_over_time(" Abs(cos) Major Axis Dot Prod. with x-axis", array_of_indicies_used, abs_cos_e_1_x_hat_Over_Time.reshape(Num_Drops, 1), ["abs(cos) angle"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot ellps stress wrt x axis:
		plts.plot_scalar_fields_over_time(" Proj Ellps Stress on x-axis", array_of_indicies_used, np.multiply(1. - abs_cos_e_1_x_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time).reshape(Num_Drops, 1), ["Stress"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot orientation wrt y axis:
		plts.plot_scalar_fields_over_time(" Abs(cos) Major Axis Dot Prod. with y-axis", array_of_indicies_used, abs_cos_e_1_y_hat_Over_Time.reshape(Num_Drops, 1), ["abs(cos) angle"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot ellps stress wrt y axis:
		plts.plot_scalar_fields_over_time(" Proj Ellps Stress on y-axis", array_of_indicies_used, np.multiply(1. - abs_cos_e_1_y_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time).reshape(Num_Drops, 1), ["Stress"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot orientation wrt z axis:
		plts.plot_scalar_fields_over_time(" Abs(cos) Major Axis Dot Prod. with z-axis", array_of_indicies_used, abs_cos_e_1_z_hat_Over_Time.reshape(Num_Drops, 1), ["abs(cos) angle"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot ellps stress wrt z axis:
		plts.plot_scalar_fields_over_time(" Proj Ellps Stress on z-axis", array_of_indicies_used, np.multiply(1. - abs_cos_e_1_z_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time).reshape(Num_Drops, 1), ["Stress"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# Plot a BUNCH of ellps orientations together:
		plts.plot_scalar_fields_over_time("Abs (cos) Major Axis Dot w axes Over Time", array_of_indicies_used, np.hstack((abs_cos_e_1_x_hat_Over_Time.reshape(Num_Drops, 1), abs_cos_e_1_y_hat_Over_Time.reshape(Num_Drops, 1), abs_cos_e_1_z_hat_Over_Time.reshape(Num_Drops, 1) )), ["x-axis", "y-axis", "z-axis"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'], y_lim_bottom_in=0., y_lim_top_in=1.)

		#hstacked_ellps_stress_proj_on_axes = np.hstack(( np.multiply(1. - abs_cos_e_1_x_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time).reshape(Num_Drops, 1), np.multiply(1. - abs_cos_e_1_y_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time).reshape(Num_Drops, 1), np.multiply(1. - abs_cos_e_1_z_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time).reshape(Num_Drops, 1) ))

		hstacked_ellps_stress_proj_on_axes = np.hstack(( Tissue_Stress_x_proj_Over_Time.reshape(Num_Drops, 1), Tissue_Stress_y_proj_Over_Time.reshape(Num_Drops, 1), Tissue_Stress_z_proj_Over_Time.reshape(Num_Drops, 1) ))


		max_ells_stress = max(hstacked_ellps_stress_proj_on_axes.flatten())
		min_ells_stress = min(hstacked_ellps_stress_proj_on_axes.flatten())

		#list_of_ellps_stress_proj_on_axes = [ np.multiply(1. - abs_cos_e_1_x_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time), np.multiply(1. - abs_cos_e_1_y_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time), np.multiply(1. - abs_cos_e_1_z_hat_Over_Time, Anis_Stress_Ellps_Max_Difs_Over_Time) ]
		list_of_ellps_stress_proj_on_axes = [ Tissue_Stress_x_proj_Over_Time, Tissue_Stress_y_proj_Over_Time, Tissue_Stress_z_proj_Over_Time ]

		print("hstacked_ellps_stress_proj_on_axes = "+str(hstacked_ellps_stress_proj_on_axes))
		print("list_of_ellps_stress_proj_on_axes = "+str(list_of_ellps_stress_proj_on_axes))
		print("min_ells_stress = "+str(min_ells_stress))
		print("max_ells_stress = "+str(max_ells_stress))

		# Plot a BUNCH of ellps proj stresses together:
		plts.plot_scalar_fields_over_time("Proj Ellps Stress on axes Over Time", array_of_indicies_used, hstacked_ellps_stress_proj_on_axes, ["x-axis", "y-axis", "z-axis"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'], y_lim_bottom_in=min_ells_stress, y_lim_top_in=float(max_ells_stress) )

		# Plot a BUNCH of ellps proj stresses together (List format):
		plts.plot_scalar_fields_over_time_From_List("Proj Ellps Stress on axes (list) Over Time", [np.arange(Num_Drops), np.arange(Num_Drops), np.arange(Num_Drops)], list_of_ellps_stress_proj_on_axes, ["x-axis", "y-axis", "z-axis"], log=False, markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'], y_lim_bottom_in=min_ells_stress, y_lim_top_in=max_ells_stress)

		# Plot a BUNCH of angles in degrees together:
		plts.plot_scalar_fields_over_time("Ellps major axes angles with axes Over Time", array_of_indicies_used, np.hstack(( (180./np.pi)*np.arccos(abs_cos_e_1_x_hat_Over_Time).reshape(Num_Drops, 1), (180./np.pi)*np.arccos(abs_cos_e_1_y_hat_Over_Time).reshape(Num_Drops, 1),  (180./np.pi)*np.arccos(abs_cos_e_1_z_hat_Over_Time).reshape(Num_Drops, 1) )), ["x-axis", "y-axis", "z-axis"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'], y_lim_bottom_in=0., y_lim_top_in=90.)

		# Plot a BUNCH of angles in pi/2 radians together:
		plts.plot_scalar_fields_over_time("Ellps major axes angles (ADJ) with axes Over Time", array_of_indicies_used, np.hstack(( (2./np.pi)*np.arccos(abs_cos_e_1_x_hat_Over_Time).reshape(Num_Drops, 1), (2./np.pi)*np.arccos(abs_cos_e_1_y_hat_Over_Time).reshape(Num_Drops, 1),  (2./np.pi)*np.arccos(abs_cos_e_1_z_hat_Over_Time).reshape(Num_Drops, 1) )), ["x-axis", "y-axis", "z-axis"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'], y_lim_bottom_in=0., y_lim_top_in=1.)


		# plot Neha angle with e_1 axis:
		if(Drop_Anal_Input_Dict['Neha_Radial_analysis'] == True):
			plts.plot_scalar_fields_over_time(" abs(Neha dot prod of e_1 with radial Animal Knot Center)", array_of_indicies_used, Neha_abs_cos_btw_rad_e1_Over_Time.reshape(Num_Drops, 1), ["abs(cos) angle"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

			plts.plot_scalar_fields_over_time(" Neha Droplet Rad Distance From Animal Knot Center", array_of_indicies_used, Neha_Drop_rad_Dist_EK.reshape(Num_Drops, 1), ["abs(cos) angle"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])


		# plot gauss-bonnet conv test:
		plts.plot_scalar_fields_over_time(" Gauss-Bonnet Conv. Test", array_of_indicies_used, Gauss_Bonnet_Test_Over_Time.reshape(Num_Drops, 1), ["rel. err."], log=True, markersize = 1., plot_type = 'o-', plot_dir = Output_Dir)

		# Examine how H0_surface (from Integral) differs from H0_Ellps, used to calculate cell stresses:
		plts.plot_scalar_fields_over_time(" H0 minus H0_ellps", array_of_indicies_used, H0_lbdv_surf_minus_Ellps_Over_Time.reshape(Num_Drops, 1), ["H0 - H0_ellps"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		## Plot ELIJAH (INPUT FIELDS): ##

		# plot 2\gamma*(H_INPUT - H0_INPUT) Alpha percentile ranges:
		plts.plot_scalar_fields_over_time(" Anis Stress (INPUT) Percentile Range Over Time", array_of_indicies_used, Anis_Stress_Input_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), ["alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisStress'])], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# plot 2\gamma*(H_INPUT-H_ellps_INPUT) Alpha percentile ranges:
		plts.plot_scalar_fields_over_time(" Anis Cell Stresses (INPUT) Alpha Percentile Range Over Time", array_of_indicies_used, AnisCellStress_Input_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), ["alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisCellStress'])], markersize = .2, plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])


		# Plot Spatial Coors, for EACH timestep from lists:
		plts.plot_scalar_fields_over_time_From_List("Spatial Corrs Over Time", Spatial_Corrs_dists_list, Spatial_Corrs_corrs_list, Spatial_Corrs_labels_list, log=False, markersize = .2, plot_type = 'o-', plot_dir = Output_Dir, bspline=False)

		# Plot NORMED Spatial Coors, for EACH timestep from lists:
		plts.plot_scalar_fields_over_time_From_List("Spatial NORMED Corrs Over Time", Spatial_Corrs_dists_list, Spatial_Normed_Corrs_corrs_list, Spatial_Corrs_labels_list, log=False, markersize = .2, plot_type = 'o-', plot_dir = Output_Dir, bspline=False, y_lim_bottom_in=-1., y_lim_top_in=1.)

		# Plot NORMED Spatial Cellular Coors, for EACH timestep from lists:
		plts.plot_scalar_fields_over_time_From_List("Spatial NORMED Cell Corrs Over Time",Cell_Corrs_dists_list, Spatial_Normed_Cell_Corrs_corrs_list, Spatial_Corrs_labels_list, log=False, markersize = .2, plot_type = 'o-', plot_dir = Output_Dir, bspline=False, y_lim_bottom_in=-1., y_lim_top_in=1.)

		# Plot NORMED Spatial Cellular Coors, for EACH timestep from lists:
		plts.plot_scalar_fields_over_time_From_List("Spatial NORMED Tissue Corrs Over Time", Tiss_Corrs_dists_list, Spatial_Normed_Tissue_Corrs_corrs_list, Spatial_Corrs_labels_list, log=False, markersize = .2, plot_type = 'o-', plot_dir = Output_Dir, bspline=False, y_lim_bottom_in=-1., y_lim_top_in=1.)


		# plot gauss-bonnet conv test:
		plts.plot_scalar_fields_over_time(" Gauss-Bonnet Radial Conv. Test", array_of_indicies_used, Gauss_Bonnet_Rad_Test_Over_Time.reshape(Num_Drops, 1), ["rel. err."], log=True, markersize = 1., plot_type = 'o-', plot_dir = Output_Dir)

		# plot avg diff in error from H_lbdv, H_Rad_lbdv:
		plts.plot_scalar_fields_over_time("H lbdv vs H_Rad Err. Test", array_of_indicies_used, H_lbdv_vs_H_Rad_Over_Time.reshape(Num_Drops, 1), ["avg diff."], log=True, markersize = 1., plot_type = 'o-', plot_dir = Output_Dir)

		# Temporal Corrs in total stress, from Radial Coors:
		plts.plot_scalar_fields_over_time(" Temporal Corrs from Radial lbdv coors", array_of_indicies_used, Temporal_Corrs_AnisStress_Rad.reshape(Num_Drops, 1), ["temp corr tau"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=False, y_lim_bottom_in=-1., y_lim_top_in=1.)

		# Temporal Corrs in cellular stress, from Radial Coors:
		plts.plot_scalar_fields_over_time(" Temporal Corrs Cell Stress (Radial lbdv coors)", array_of_indicies_used, Temporal_Corrs_Cell_Stress_Rad.reshape(Num_Drops, 1), ["temp corr tau"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=False, y_lim_bottom_in=-1., y_lim_top_in=1.)

		# Temporal Corrs in tissue stress, from Radial Coors:
		plts.plot_scalar_fields_over_time(" Temporal Corrs Tissue Stress (Radial lbdv coors)", array_of_indicies_used, Temporal_Corrs_Tissue_Stress_Rad.reshape(Num_Drops, 1), ["temp corr tau"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=False, y_lim_bottom_in=-1., y_lim_top_in=1.)


		# Plot a BUNCH of fields together:
		plts.plot_scalar_fields_over_time("Stress Comparisons Over Time", array_of_indicies_used, np.hstack((Anis_Stress_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), AnisCellStress_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), Anis_Stress_Ellps_Max_Difs_Over_Time.reshape(Num_Drops, 1) )), ["2\gamma(H - H_0): alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisStress']), "2\gamma(H - H_e) : alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisCellStress']), "max. Ellps. Stress Ans."], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# Plot a BUNCH of INPUT fields together:
		plts.plot_scalar_fields_over_time("Stress (INPUTS) Comparisons Over Time", array_of_indicies_used, np.hstack((Anis_Stress_Input_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), AnisCellStress_Input_Alpha_Percentile_Dif_Over_Time.reshape(Num_Drops, 1), Anis_Stress_Ellps_Max_Difs_Over_Time.reshape(Num_Drops, 1) )), ["2\gamma(H_in - H_0_in): alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisStress']), "2\gamma(H_in - H_e) : alpha = "+str(Drop_Anal_Input_Dict['alpha_percentile_excl_AnisCellStress']), "max. Ellps. Stress Ans."], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=Drop_Anal_Input_Dict['smoothBSpline'])

		# Plot a BUNCH of Gauss-Bonnet Test fields together:
		plts.plot_scalar_fields_over_time("Comparing Gauss-Bonnet Tests", array_of_indicies_used, np.hstack((Gauss_Bonnet_Test_Over_Time.reshape(Num_Drops, 1), Gauss_Bonnet_Rad_Test_Over_Time.reshape(Num_Drops, 1) )), ["lbdv ellps coors", "lbdv rad coors"], log=True, markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=False)

		# Plot a BUNCH of H0 vals together:
		plts.plot_scalar_fields_over_time("H0 Comparisons Over Time", array_of_indicies_used, np.hstack((H0_From_Vol_Int.reshape(Num_Drops, 1), H0_From_Area_Int.reshape(Num_Drops, 1), H0_From_Ellipsoid.reshape(Num_Drops, 1), H0_From_avg_lbdv_pts.reshape(Num_Drops, 1), H0_From_avg_seg_pts.reshape(Num_Drops, 1) )), ["H0_From_Vol_Int", "H0_From_Area_Int", "H0_From_Ellipsoid", "H0_From_avg_lbdv_pts", "H0_From_avg_seg_pts"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=False)

		# same, but scale to y_lim_bottom_in=0.:
		plts.plot_scalar_fields_over_time("H0 Comparisons Over Time (from 0)", array_of_indicies_used, np.hstack((H0_From_Vol_Int.reshape(Num_Drops, 1), H0_From_Area_Int.reshape(Num_Drops, 1), H0_From_Ellipsoid.reshape(Num_Drops, 1), H0_From_avg_lbdv_pts.reshape(Num_Drops, 1), H0_From_avg_seg_pts.reshape(Num_Drops, 1) )), ["H0_From_Vol_Int", "H0_From_Area_Int", "H0_From_Ellipsoid", "H0_From_avg_lbdv_pts", "H0_From_avg_seg_pts"], markersize = 1., plot_type = 'o-', plot_dir = Output_Dir, bspline=False, y_lim_bottom_in=0.)

		# Save .mat file output, so it can be read in MATLAB:
		Dict_for_Matfile = {}
		#! NAMES CANNOT HAVE SPACES !#:
		Dict_for_Matfile['Drop_Indicies_Used'] = array_of_indicies_used.flatten()
		Dict_for_Matfile['H0_From_Area_Int'] = H0_From_Area_Int.flatten()
		Dict_for_Matfile['Number_of_Segmeted_Pts_Input'] = Num_Dropets_Seg_Pts_Over_Time.flatten()
		Dict_for_Matfile['Volume_Estimates_of_Droplets'] = Volumes_Measured_Over_Time.flatten()
		Dict_for_Matfile['Gauss_Bonnet_Integral_Test'] = Gauss_Bonnet_Test_Over_Time.flatten()
		Dict_for_Matfile['Anis_Stress_Drop_Inter_Percentile_Range'] = Anis_Stress_Alpha_Percentile_Dif_Over_Time.flatten()
		Dict_for_Matfile['Anis_Stress_Cells_Inter_Percentile_Range'] = AnisCellStress_Alpha_Percentile_Dif_Over_Time.flatten()
		Dict_for_Matfile['Max_Anis_Stress_LS_Ellipsoid'] = Anis_Stress_Ellps_Max_Difs_Over_Time.flatten()
		Dict_for_Matfile['Abs_Cos_Orientation_w_Final_Axis'] = Abs_Cos_Orient_Over_Time.flatten()
		Dict_for_Matfile['abs_cos_e_1_x_hat_Over_Time'] = abs_cos_e_1_x_hat_Over_Time.flatten()
		Dict_for_Matfile['abs_cos_e_1_y_hat_Over_Time'] = abs_cos_e_1_y_hat_Over_Time.flatten()
		Dict_for_Matfile['abs_cos_e_1_z_hat_Over_Time'] = abs_cos_e_1_z_hat_Over_Time.flatten()
		Dict_for_Matfile['Tissue_Stress_x_proj_Over_Time'] = Tissue_Stress_x_proj_Over_Time.flatten()
		Dict_for_Matfile['Tissue_Stress_y_proj_Over_Time'] = Tissue_Stress_y_proj_Over_Time.flatten()
		Dict_for_Matfile['Tissue_Stress_z_proj_Over_Time'] = Tissue_Stress_z_proj_Over_Time.flatten()

		if(Drop_Anal_Input_Dict['Neha_Radial_analysis'] == True):
			Dict_for_Matfile['Neha_abs_cos_btw_rad_e1_Over_Time'] = Neha_abs_cos_btw_rad_e1_Over_Time.flatten()
			Dict_for_Matfile['Neha_angle_deg_rad_e1_Over_Time'] = np.arccos(Dict_for_Matfile['Neha_abs_cos_btw_rad_e1_Over_Time'])*180./np.pi			
			Dict_for_Matfile['Neha_rad_Dist_from_EK_Over_Time'] = Neha_Drop_rad_Dist_EK.flatten()
			Dict_for_Matfile['Neha_DropCent_to_EK_vecs'] = Neha_DropCent_to_EK_vecs
			print("Dict_for_Matfile['Max_Anis_Stress_LS_Ellipsoid'] = "+str(Dict_for_Matfile['Max_Anis_Stress_LS_Ellipsoid']))
			print("Dict_for_Matfile['Neha_abs_cos_btw_rad_e1_Over_Time'] = "+str(Dict_for_Matfile['Neha_abs_cos_btw_rad_e1_Over_Time']))
			print("Dict_for_Matfile['Neha_angle_deg_rad_e1_Over_Time'] = "+str(Dict_for_Matfile['Neha_angle_deg_rad_e1_Over_Time']))
			print("Dict_for_Matfile['Anis_Stress_Cells_Inter_Percentile_Range'] = "+str(Dict_for_Matfile['Anis_Stress_Cells_Inter_Percentile_Range']))
			print("Dict_for_Matfile['Neha_rad_Dist_from_EK_Over_Time'] = "+str(Dict_for_Matfile['Neha_rad_Dist_from_EK_Over_Time']))
			print("Dict_for_Matfile['Neha_DropCent_to_EK_vecs']  = "+str( Dict_for_Matfile['Neha_DropCent_to_EK_vecs']  ))
			
		# LS_Elloid_Info:
		Dict_for_Matfile['LS_Ellps_axes_vecs'] = LS_Ellps_axes_vecs
		Dict_for_Matfile['LS_Ellps_Inv_Rot_mats'] = LS_Ellps_Inv_Rot_mats
		Dict_for_Matfile['LS_Ellps_center_vecs'] = LS_Ellps_center_vecs
		Dict_for_Matfile['Tension_gamma_val'] = Drop_Anal_Input_Dict['Tension_gamma_val']
		Dict_for_Matfile['Pixel_Size_Microns'] = Drop_Anal_Input_Dict['Pixel_Size_Microns']
		matfile_file_path = os.path.join(Output_Dir, "Output_For_MATLAB.mat")

		# CORRS INFO:
		Dict_for_Matfile['Temporal_Corrs_AnisStress_Rad'] = Temporal_Corrs_AnisStress_Rad
		Dict_for_Matfile['Temporal_Corrs_Cell_Stress_Rad'] = Temporal_Corrs_Cell_Stress_Rad
		Dict_for_Matfile['Temporal_Corrs_Tissue_Stress_Rad'] = Temporal_Corrs_Tissue_Stress_Rad

		Dict_for_Matfile['Spatial_Corrs_dists_list'] = Spatial_Corrs_dists_list
		Dict_for_Matfile['Spatial_Normed_Corrs_corrs_list'] = Spatial_Normed_Corrs_corrs_list
		Dict_for_Matfile['Spatial_Corrs_labels_list'] = Spatial_Corrs_labels_list
		Dict_for_Matfile['Cell_Corrs_dists_list'] = Cell_Corrs_dists_list
		Dict_for_Matfile['Spatial_Normed_Cell_Corrs_corrs_list'] = Spatial_Normed_Cell_Corrs_corrs_list
		Dict_for_Matfile['Tiss_Corrs_dists_list'] = Tiss_Corrs_dists_list
		Dict_for_Matfile['Spatial_Normed_Tissue_Corrs_corrs_list'] = Spatial_Normed_Tissue_Corrs_corrs_list

		#print("matfile_file_path = "+str(matfile_file_path))
		#print("Dict_for_Matfile = "+str(Dict_for_Matfile))

		savemat(matfile_file_path, Dict_for_Matfile)
