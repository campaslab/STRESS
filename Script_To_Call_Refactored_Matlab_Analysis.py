#! Script to test refactored Batch_Droplet_Class_Analysis.py
import os, sys
from scipy.io import savemat # for writing .mat file, for Elijah's code to read
from scipy.io import loadmat # for reading .mat file
import Batch_Droplet_Class_Analysis as BDCA

MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
MAT_Files_Dir = os.path.join(MY_DIR, 'Mat_Files_Load') #where we will load this data from 

Drop_Input_Dict = {} # dictionary we need to use Droplet_Analysis objects
Drop_Input_Dict['SubFolder_We_Load_From'] = 'Elijah_dozen_seq_8_28_20' #'24-bud2 111920' #'16 101520' 

#'Elijah_dozen_seq_FOR_PAPER_8_07_20_First_21_Frames' #'Elijah_dozen_seq_FOR_PAPER_8_07_20_First_4_Frames' #'Elijah_dozen_seq_FOR_PAPER_8_07_20_First_Frame' # 'Ellipsoid_a_in_8pt0_b_in_10pt0_c_in_12pt0_5810_pts_bumps_freq_0pt1' #

# DATASETS to load:
'''
!!!! ELIJAH 8/07/20 IFT = 3.33 !!!!

#'2 070920' #'3 072820' #'4 073020' #'5 080520' #'6 080620' #'7 081820' #'8 082020' #'10 092320' #'11 092420' #'15 101420' #'16 101520' #'17 102120 (replaced bad frame 1)' #'18 102720' #'19 102820 (replaced bad frame 1)' #'20 110220' #'21 110520' #'22 111620_no carbogen' #'24-bud1 111920' #'24-bud2 111920' #'25 112320'   

!!!! NEHA TENSION = 5.17 !!!!

#'Elijah_dozen_seq_FOR_PAPER_8_07_20_First_21_Frames' # 'Elijah_dozen_seq_FOR_PAPER_8_07_20_First_4_Frames' 
#'Carlos_9_8_20_Large_Vol_Dev_Part1' #'Carlos_9_8_20_Large_Vol_Dev_Part2' #
#'Elijah_dozen_seq_8_22_20' #'Elijah_dozen_seq_8_21_20' #'Elijah_dozen_seq_8_15_20' #'Elijah_dozen_seq_8_14_20' #'Elijah_dozen_seq_8_08_20' #'Elijah_dozen_seq_8_07_20' #'Elijah_dozen_seq_8_06_20' #'Elijah_dozen_seq_8_01_20' #'Elijah_dozen_seq_7_31_20' #'Elijah_dozen_seq_7_25_20' #'Elijah_dozen_seq_7_24_20' 
#'Elijah_Reseg_TimeStep_Test_Data' #'Elijah_TimeStep_Test_Data' 
#'Carlos_9_15_20_Oil_PFOB_Drop_3' #'Carlos_9_15_20_Oil_7300_Drop_3' # 'Carlos_9_15_20_Oil_7700_Drop_3' # 'Carlos_10_1_20_Oil_7500_Drop_3'
# 'Carlos_10_6_20' #
# 'Ellipsoid_a_in_6pt0_b_in_10pt0_c_in_14pt0_5810_pts_Rot_freq_0pt1' # Ellipsoid_a_in_6pt0_b_in_10pt0_c_in_14pt0_5810_pts_Wave_freq_0pt1 #
# 'Elijah_7_24_20_Just_frame_66' # 'Elijah_dozen_seq_7_24_20_Replace_Bad_Frames'
#'Neha_070120_epi_test' #'Neha_070920_mes' # 'Neha_072820_epi_not_moved_after_3hr' # 'Neha_Elijah_10_15_20_for_grant_fig'
'''

Drop_Input_Dict['MatFile_filepath'] = os.path.join(MAT_Files_Dir, Drop_Input_Dict['SubFolder_We_Load_From']) # Test input dir
Drop_Input_Dict['Pixel_Size_Microns'] = 2.076 # number of microns per pixel length, so we can get physical values for stress (puts curvature in units of 1/microns)
Drop_Input_Dict['Tension_gamma_val'] = 5.17 #3.33 #1. # \gamma value, need to multiply by 2\gamma to get stress
Drop_Input_Dict['Use_PyCompadre'] = False
#Drop_Input_Dict['PyCompadre_p_order'] = int(2)
Drop_Input_Dict['Plot_Vtu_Outputs'] = True
Drop_Input_Dict['Output_Plot_Subfolder_Name'] = Drop_Input_Dict['SubFolder_We_Load_From']+"_OUTPUT" #'Test_of_Refactor_Plots' # directory for plotting outputs
Drop_Input_Dict['deg_lbdv_fit'] = 5 #14 #20 # degree of lebedev quad we use for fitting shape in SPB (Ellipsoidal Coors)
Drop_Input_Dict['MAX_lbdv_fit_PTS'] = False #True #False # if true, use 5810 points, if not use min hyper-interpolations points needed (default)
Drop_Input_Dict['deg_lbdv_Vol_Int'] = Drop_Input_Dict['deg_lbdv_fit'] #11
Drop_Input_Dict['MAX_lbdv_vol_PTS'] = Drop_Input_Dict['MAX_lbdv_fit_PTS'] #False
Drop_Input_Dict['deg_fit_Ellipsoid_Deviation_Analysis']  = Drop_Input_Dict['deg_lbdv_fit'] # degree for analyzing modes of deviation from Ellipsoid
Drop_Input_Dict['alpha_percentile_excl_AnisStress'] = .05 #2\gamma (H - H_Ellips), percent of extreme values we exclude on either end for analyzing stats
Drop_Input_Dict['alpha_percentile_excl_AnisCellStress'] = Drop_Input_Dict['alpha_percentile_excl_AnisStress'] #.05 # 2\gamma (H - H_Ellips), percent of extreme values we exclude on either end for analyzing stats
Drop_Input_Dict['alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin'] = Drop_Input_Dict['alpha_percentile_excl_AnisStress'] #.05
Drop_Input_Dict['Corr_Start_Dist'] = int(0) #int(3) # distance at which we start looking at correlations (since there is a noisy spike at dist = 0)
Drop_Input_Dict['Neha_Radial_analysis'] = False #True #False #True # if we do radial analysis of e_1 mode for Neha
Drop_Input_Dict['Calc_Spat_Temp_Corrs'] = False #True
Drop_Input_Dict['smoothBSpline'] = True # whether or not we smooth CERTAIN PLOTS (if we have enough pts)

test_matfile_file_path = os.path.join(MY_DIR, "Test_Input_Dict.mat")
savemat(test_matfile_file_path, Drop_Input_Dict)

'''
Test_Dict_Loaded = loadmat(test_matfile_file_path)
print("Test_Dict_Loaded['smoothBSpline'] = "+str(bool(Test_Dict_Loaded['smoothBSpline'][0,0])))
print("Test_Dict_Loaded['MAX_lbdv_fit_PTS'] = "+str(bool(Test_Dict_Loaded['MAX_lbdv_fit_PTS'][0,0])))
print("Test_Dict_Loaded['Tension_gamma_val'] = "+str(Test_Dict_Loaded['Tension_gamma_val'][0,0]))
B = BDCA.Batch_Droplet_Class_Analysis(Test_Dict_Loaded)
'''

BDCA.Load_Dict_From_Matfile(test_matfile_file_path)

#Test_load_dict = BDCA.Load_Dict_From_Matfile(test_matfile_file_path)
#B = BDCA.Batch_Droplet_Class_Analysis(Test_load_dict)

#B = BDCA.Batch_Droplet_Class_Analysis(Drop_Input_Dict)
