import os, sys
from scipy.io import savemat # for writing .mat file, for Elijah's code to read
from scipy.io import loadmat # for reading .mat file
import Batch_Droplet_Class_Analysis as BDCA

MY_DIR  = os.path.realpath(os.path.dirname(__file__)) 
test_matfile_file_path = os.path.join(MY_DIR, "Input_Dict.mat")

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
