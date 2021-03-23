%% AnalyzeDeluxe3D
%% Created Feb 5, 2021 by Elijah Shelton
%% Last modified Feb 9, 2021 by Elijah Shelton
%% Elijah Shelton is a Ph.D. Candidate at UC Santa Barbara in the Campas Group
%% Contact: eshelton@ucsb.edu
%% The goal of this script is to analyse the geometry of droplets starting from a 3D or 4D tif-stack.
%% STEP 1: Before running this script, crop your tifs.
% Before running this script, I recommend openning your image in FIJI. Crop
% the volume such that the droplet is approximately one radius from the
% boundary at every timestep. If your original image contained multiple
% channels, select only the channel containing the droplet signal. Save
% this single-channel, cropped, 3D or 4D image as a tif file.
%% STEP 2: Gather your parameters
% Make sure you know the following:
% a) Size of each pixel (in microns)
% b) Size of each z-step (in microns)
% c) Whether the surface or the bulk of the drop is labeled.
% d) The interfacial tension of the drop in mN/m
%% STEP 3: Run the Script
% Click 'Run' on the toolbar, or type the name of this file
% ('AnalyseBasic3D') into the Command Window.
% Follow all prompts.
%% Clean up the workspace
clear % clear variables from workspace
close all % close all figures
clc % clear all text from command window
%% Python Check
%% Run PYTHON Analysis
pythonCall = {'python3','py3','python','py'};
pyCallIdx = 0;
for n = 1:length(pythonCall)
    testInputString = [pythonCall{n},' test_import.py'];
    errOut = system(testInputString);
    if errOut == 0
        pyCallIdx = n;
        break
    end
end
if ~pyCallIdx
    maxIter = 3;
    iterNum = 0;
    errOut = 1;
    while errOut
        iterNum = iterNum + 1;
        fprintf('\nHmm...I''m having trouble running python 3 on your machine.\n')
        fprintf('\nWhat do you normally use to run python 3? (i.e python3, py, etc.)\n')
        pythonCall{5} = input('>> ','s');
        pyCallIdx = 5;
        errOut = system(testInputString);
        if iterNum == maxIter
           pyCallIdx = 0;
           fprintf('\nMax number of attempts made.\n')
           break 
        end
    end
end
if pyCallIdx
    fprintf('\nGood!\nWe can use %s to call python3.\n',pythonCall{pyCallIdx})
    fprintf('You also have all modules installed.\n')
else
    fprintf('\nProblem detected with installation of python 3 or associated modules.\n')
    fprintf('\nCheck command window for details.\n')
    error('Problem with Python 3 or associated modules.')
end

%% Turn off certain warnings (feel free to edit)
warning('off','MATLAB:imagesci:tiffmexutils:libtiffErrorAsWarning')
warning('off','MATLAB:imagesci:tiffmexutils:libtiffWarning')
fprintf('Segmenting drop...\n')
segObj = seg3D;
%segObj = OnionSkin;
TimelapseObj = segObj.Timelapse();
clear segObj btn1 btn2 defBtn rsp

%% Run PYTHON Analysis
callMeString = [pythonCall{pyCallIdx},' Call_Me_At_The_End.py'];
errOut = system(callMeString);


%% Results
fprintf('\nYour results are saved in the following files:\n')
for n = 1:length(TimelapseObj)
    fprintf('%d. %s\n',n,TimelapseObj{n}.Path2Mat)
end
fprintf('\nThese files are located in the following directory:\n');
fprintf('\n%s\n',TimelapseObj{n}.Path2MatDir)
DictIn = load(TimelapseObj{1}.Path2InputDictMat);
path2DeluxeAnalysis = ['Outputs',filesep,DictIn.SubFolder_We_Load_From];
fprintf('Your DELUXE outputs are located in %s\n',path2DeluxeAnalysis)
%% 