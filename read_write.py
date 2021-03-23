import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

# =========================================
#
# Author: Paul J. Atzberger (copied by BJG to modify Dynkin Data)
#
# Interface for reading and writing VTK 
# data sets.
#
# -----------------------------------------

def write_vtp(vtp_filename, points, fieldData_list,flagVerboseLevel=0,flagFieldDataMode=None):
  ambient_num_dim =3; # assumed data is embedded in R^3

  if flagFieldDataMode == None:
    if isinstance(fieldData_list, list) == True:
      flagFieldDataMode='list';
    elif isinstance(fieldData_list, dict) == True:
      flagFieldDataMode='dict';

  # Check parameter types are correct given the flags
  flagFieldTypeErr = False;
  if (flagFieldDataMode == 'list') and (isinstance(fieldData_list, list) == False):
    flagFieldTypeErr = True;
    type_str = str(list);
  elif (flagFieldDataMode == 'dict') and (isinstance(fieldData_list, dict) == False):
    flagFieldTypeErr = True;
    type_str = str(dict);

  if flagFieldTypeErr == True:
    err_s = "";
    err_s += "Expected that the fieldData_list is of type '%s'. \n"%type_str;
    err_s += "May need to adjust and set the flagFieldDataMode. \n";
    err_s += "flagFieldDataMode = %s\n"%str(flagFieldDataMode);
    err_s += "type(fieldData_list) = %s\n"%str(type(fieldData_list));
    raise Exception(err_s);

  # Output a VTP file with the fields
  #
  # record the data for output
  #N_list = np.shape(ptsX)[0];  
  vtpData = vtk.vtkPolyData();

  # check format of the points array
  #n1 = np.size(points_data,0);
  #if (n1 == 0): # we assume array already flat
  #  # nothing to do
  #else:
  #  points = points_data.T.flatten();

  # setup the points data
  Points = vtk.vtkPoints();
  #numPoints = int(len(points)/ambient_num_dim);
  numPoints=np.size(points,1);
  for I in range(numPoints): 
    Points.InsertNextPoint(points[0,I], points[1,I], points[2,I]);
  vtpData.SetPoints(Points);
 
  # Get data from the vtu object
  #print("Getting data from the vtu object.");
  #nodes_vtk_array = vtuData.GetPoints().GetData();
  #ptsX            = vtk_to_numpy(nodes_vtk_array);

  # -- setup data arrays
  if fieldData_list != None: # if we have field data

    numFields = len(fieldData_list);
    if flagVerboseLevel==1:
      print("numFields = " + str(numFields));

    if flagFieldDataMode=='list':
      f_list = fieldData_list;
    elif flagFieldDataMode=='dict': 
      # convert dictionary to a list
      f_list=[];
      for k, v in fieldData_list.items():
        f_list.append(v);

    for fieldData in f_list:
      #print("fieldData = " + str(fieldData));
      fieldName = fieldData['fieldName'];
      if flagVerboseLevel==1:
        print("fieldName = " + str(fieldName));
      fieldValues = fieldData['fieldValues'];
      NumberOfComponents = fieldData['NumberOfComponents'];

      if (NumberOfComponents == 1):
        N_list     = len(fieldValues);
        if flagVerboseLevel==1:
          print("N_list = " + str(N_list));
        atzDataPhi = vtk.vtkDoubleArray();
        atzDataPhi.SetNumberOfComponents(1);
        atzDataPhi.SetName(fieldName);
        #print("fieldName = "+str(fieldName))
        atzDataPhi.SetNumberOfTuples(N_list);
        for I in np.arange(0,N_list):
          #print("I = "+str(I)+", fieldValues[I] = "+str(fieldValues[I]))
          atzDataPhi.SetValue(I,fieldValues[I]);
        #vtpData.GetPointData().SetScalars(atzDataPhi);
        vtpData.GetPointData().AddArray(atzDataPhi);
      elif (NumberOfComponents == 3): 
        #print(fieldValues);
        #print(fieldValues.shape);
        N_list   = fieldValues.shape[1];
        if flagVerboseLevel==1:
          print("N_list = " + str(N_list));
        atzDataV = vtk.vtkDoubleArray();
        atzDataV.SetNumberOfComponents(3);
        atzDataV.SetName(fieldName);
        atzDataV.SetNumberOfTuples(N_list);
        for I in np.arange(0,N_list):
          atzDataV.SetValue(I*ambient_num_dim + 0,fieldValues[0,I]);
          atzDataV.SetValue(I*ambient_num_dim + 1,fieldValues[1,I]);
          atzDataV.SetValue(I*ambient_num_dim + 2,fieldValues[2,I]);
        #vtpData.GetPointData().SetVectors(atzDataV);
        vtpData.GetPointData().AddArray(atzDataV);

      else:
        #print("ERROR: " + error_code_file + ":" + error_func);
        s = "";
        s += "NumberOfComponents invalid. \n";
        s += "NumberOfComponents = " + str(NumberOfComponents);
        raise Exception(s);

        #exit(1);

  #vtuData.GetPointData().SetVectors(atzDataVec);
  #vtuData.GetPointData().AddArray(atzDataScalar1);
  #vtuData.GetPointData().AddArray(atzDataScalar2);
  #vtuData.GetPointData().AddArray(atzDataVec1);
  #vtuData.GetPointData().AddArray(atzDataVec2);

  # write the XML file
  writerVTP = vtk.vtkXMLPolyDataWriter();
  writerVTP.SetFileName(vtp_filename);
  writerVTP.SetInputData(vtpData);
  writerVTP.SetCompressorTypeToNone(); # help ensure ascii output (as opposed to binary)
  writerVTP.SetDataModeToAscii(); # help ensure ascii output (as opposed to binary)
  writerVTP.Write();   
  #writerVTP.Close();

# Loads a VTP file into python data structures.
#
# fieldNames: collection of fields to load, if None loads all. 
#
def read_vtp(vtp_filename,fieldNames=None,flagVerboseLevel=0,flagFieldDataMode='dict'):
  vtpData = {};
  metaData = {};
    
  metaData['vtp_filename'] = vtp_filename;

  # Load the data
  reader = vtk.vtkXMLPolyDataReader();
  reader.SetFileName(vtp_filename);
  reader.Update();

  # Collect the coordinates of the data.
  nodes_vtk_array= reader.GetOutput().GetPoints().GetData();
  X              = vtk_to_numpy(nodes_vtk_array);
  vtpData['points'] = X.T;

  # Collect data from the vtk file.
  # Loop over the fields and collect the field data.
  p = reader.GetOutput().GetPointData();

  num_fields = p.GetNumberOfArrays();

  if flagFieldDataMode=='list':
    fieldData_list = [];
  elif flagFieldDataMode=='dict':
    fieldData_list = {};

  for k in np.arange(0,num_fields):
    fieldName = p.GetArrayName(k);

    f = {};
    vtk_array      = reader.GetOutput().GetPointData().GetArray(fieldName);
    f['fieldName'] = fieldName;
    V              = vtk_to_numpy(vtk_array);
    #u,v,w          = V[:,0], V[:,1], V[:,2];
    f['fieldValues'] = V.T;
    r = len(np.shape(f['fieldValues'])); # number of dimensions
    if r <= 1:
      f['NumberOfComponents'] = 1;
    else:
      f['NumberOfComponents'] = np.size(f['fieldValues'],0);

    if flagFieldDataMode=='list':
      fieldData_list.append(f);
    elif flagFieldDataMode=='dict':
      fieldData_list[fieldName] = f;  

  vtpData['flagFieldDataMode'] = flagFieldDataMode;
  vtpData['fieldData_list'] = fieldData_list;  
  
  vtpData['metaData'] = metaData;

  return vtpData;

# -----------------------------------------
#
# =========================================


