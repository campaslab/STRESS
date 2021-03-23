%% ASSEMBLE EK TIF FILES
% WRITTEN ON 11/1/2020 by ELIJAH R SHELTON
% Based on ASSEMBLE TIMELAPSE TIF FILES Written by Elijah R Shelton on July 28 2020 & Modified by Elijah R Shelton on September 2 2020
%
% AIM: The purpose of this script is to prepare tif z stacks from the
% original multichannel files.
% The tif sequences should be ready to be reconstructed using seg3D
% Additionally, MIP tif sequences will also be prepared for reconstruction
% with seg2D, or simply visualization.
close all
clear
clc

%% STEP 1: 
% SPECIFY PATHS TO TIFS TO LOAD and TIFS TO SAVE
[paths2TifFiles,paths2Crop3DTifs,paths2MIPZTifs,paths2CropInfo] = getPaths2Tifs();
%% STEP 2:
% LOOP THROUGH EACH PATH
numOfTifs = numel(paths2TifFiles);
for n = 1:numOfTifs
    %% STEP 2a.
    % GET IMAGES AND IMFINFO PREVIEW
    tic
    [im_stack_ch1,im_stack_ch2,im_info] = getImagesAndInfo(paths2TifFiles{n},'full');
    toc
    %% STEP 2b.
    % DETERMINE WHICH CHANNEL CONTAINS DROP
    [im_dropCh,im_otherCh,dropChIndex] = specifyDropChannel(im_stack_ch1,im_stack_ch2);
    %% STEP 2c.
    % CLEAN UP DROPLET CHANNEL
    [im_dropCh_clean] = cleanUpDropChannel(im_dropCh,im_otherCh);
    %% STEP 2d.
    % CREATE MIP
    im_dropCh_clean_MIP = max(im_dropCh_clean,[],3);
    %% STEP 2e.
    % CROP 3D
    [im_drop_crop,cropInfo] = crop3D(im_dropCh_clean,im_dropCh_clean_MIP);    
    
    %% STEP 2g.
    % SAVE 3D CROP AS TIFF STACKS (W/ META DATA)
    path2Crop3DTif = paths2Crop3DTifs{n};
    saveTif(im_drop_crop,path2Crop3DTif,im_info);
    % SAVE MIP_Z AS TIFF STACKS (W/ META DATA)
    path2MIPZTif = paths2MIPZTifs{n};
    saveTif(im_dropCh_clean_MIP,path2MIPZTif,im_info);
    %% SAVE CROP INFO
    path2CropInfo = paths2CropInfo{n};
    saveCropInfo(path2CropInfo,cropInfo)
    
end
% SUPPORT FUNCTIONS
function[] = saveCropInfo(path2CropInfo,cropInfo)
rowMin = cropInfo.rMin;
rowMax = cropInfo.rMax;
colMin = cropInfo.cMin;
colMax = cropInfo.cMax;
save(path2CropInfo,'rowMin','rowMax','colMin','colMax');
end
function[] = saveTif(im,path2Tif,im_info)
[~,~,depth] = size(im);
Resolution = [im_info(1).XResolution, im_info(1).YResolution];
ImageDescription = im_info(1).ImageDescription;
imwrite(im(:,:,1),path2Tif,'Resolution',Resolution);
tic
for n = 2:depth    
    imwrite(im(:,:,n),path2Tif,'Resolution',Resolution,...
        'WriteMode','append');
    elapsedTime = toc;
    if elapsedTime > 0.5
        tic
        fprintf('Saving %d of %d\n',n,depth)
    end
end
t = Tiff(path2Tif,'r+');
setTag(t,'ImageDescription',ImageDescription);
close(t)
end


function[Im_crop,cropInfo] = crop3D(Im,Im_mip)
%[rows,cols] = size(Im_mip);
fprintf('Draw rectangle to CROP!\n')
figure(1)
imshow(Im_mip,[]);
polyrect = drawrectangle();
col = polyrect.Position(1);
row = polyrect.Position(2);
width = polyrect.Position(3);
height = polyrect.Position(4);
%Im_crop = Im_mip(row:row+height,col:col+width);
rMin = row;
cMin = col;
rMax = row+height;
cMax = col+width;
cropInfo.rMin = rMin;
cropInfo.cMin = cMin;
cropInfo.rMax = rMax;
cropInfo.cMax = cMax;
Im_crop = Im(rMin:rMax,cMin:cMax,:);
depth = size(Im_crop,3);
z_slice = round(depth/2);
imshow(Im_crop(:,:,z_slice),[])
% 
% Im_thresh = median(Im_crop(:));
% [a,b,xCenter,yCenter,~] = getEllipseFromImage(Im_mip,Im_thresh);
% rMin = uint16(xCenter-1.5*max([a,b]));
% cMin = uint16(yCenter-1.5*max([a,b]));
% rMax = uint16(xCenter+1.5*max([a,b]));
% cMax = uint16(yCenter+1.5*max([a,b]));
% rMin = max([1,rMin]);
% cMin = max([1,cMin]);
% rMax = min([rows,rMax]);
% cMax = min([cols,cMax]);
% cropInfo.rMin = rMin;
% cropInfo.cMin = cMin;
% cropInfo.rMax = rMax;
% cropInfo.cMax = cMax;
% Im_crop = Im(rMin:rMax,cMin:cMax,:);
end


function[im_dropCh_clean] = cleanUpDropChannel(im_dropCh,im_otherCh)
numOfFactors = 5;
bleedFactors = logspace(-0.25,0.25,numOfFactors);
depth = size(im_dropCh,3);
z_slices = floor(depth/2) + (-1:2);
im_cat = [];
close all
for n = 1:numOfFactors
    bleedFactor = bleedFactors(n);
    im_dropCh_clean_z_slices = double(im_dropCh(:,:,z_slices)) - bleedFactor*double(im_otherCh(:,:,z_slices));
    im_dropCh_clean_z_slices(im_dropCh_clean_z_slices<0) = 0;
    im_dropCh_clean_z_slices = uint16(im_dropCh_clean_z_slices);
    im_dropCh_clean_z_slice_4x4bin = binImage3D(im_dropCh_clean_z_slices,4,4);
    im_cat = [im_cat,im_dropCh_clean_z_slice_4x4bin];
    listString{n} = num2str(n);
    fprintf('Processing %d of %d\n',n,numOfFactors);
end
imshow(im_cat,[]);
promptString = 'Select the best image (1-5 from left to right)';
list_rsp = listdlg('PromptString',promptString,'ListString',listString);
bleedFactor = bleedFactors(list_rsp);
im_dropCh_clean = uint16(im_dropCh) - uint16(bleedFactor*double(im_otherCh));
im_dropCh_clean(im_dropCh_clean<0) = 0;
close all
end

function [im3DOut] = binImage3D(im3DIn,binFactorXY,binFactorZ)
[rowsIn,colsIn,pagesIn] = size(im3DIn);
rowsOut = floor(rowsIn/binFactorXY);
colsOut = floor(colsIn/binFactorXY);
pagesOut = floor(pagesIn/binFactorZ);
switch class(im3DIn)
    case 'uint8'
        im_binned_pages = uint8(zeros(rowsOut,colsOut,binFactorZ));
        im3DOut = uint8(zeros(rowsOut,colsOut,pagesOut));
    case 'uint16'
        im_binned_pages = uint16(zeros(rowsOut,colsOut,binFactorZ));
        im3DOut = uint16(zeros(rowsOut,colsOut,pagesOut));
    otherwise
        error('Did not expect class %s.',class(im3DIn));
end
for p_in = 1:pagesIn
    im2DIn = im3DIn(:,:,p_in);
    p_mod = mod(p_in-1,binFactorZ)+1;
    im_binned2D = binImage2D(im2DIn,binFactorXY);
    im_binned_pages(:,:,p_mod) = im_binned2D;
    if p_mod==binFactorZ
        p_out = floor((p_in-1)./binFactorZ)+1;
        im3DOut(:,:,p_out) = sum(im_binned_pages,3)/binFactorZ;
        im_binned_pages(:) = 0;
    end
    
end

end

function [imOut] = binImage2D(imIn,binFactor)
[rowsIn,colsIn] = size(imIn);
rowsOut = floor(rowsIn/binFactor);
colsOut = floor(colsIn/binFactor);
switch class(imIn)
    case 'uint8'
        imOut = uint8(zeros(rowsOut,colsOut));
    case 'uint16'
        imOut = uint16(zeros(rowsOut,colsOut));
    otherwise
        error('Did not expect class %s.',class(imIn));
end
for m = 1:rowsOut
    for n = 1:colsOut
        binRows = binFactor*(m-1) + (1:binFactor);
        binCols = binFactor*(n-1) + (1:binFactor);
        binInts = imIn(binRows,binCols);
        imOut(m,n) = mean(binInts(:));
    end
end
end

function[im_dropCh,im_otherCh,dropChIndex] = specifyDropChannel(im_stack_ch1,im_stack_ch2)
figure(1)
im_MIP_ch1 = max(im_stack_ch1,[],3);
im_MIP_ch2 = max(im_stack_ch2,[],3);
im_cat = [im_MIP_ch1,im_MIP_ch2];
imshow(im_cat,[])
ch_select = questdlg('Which image contains the drop?','Choose drop channel','Left', 'Right', 'Left');
switch ch_select
    case 'Left'
        dropChIndex = 1;
        im_dropCh = im_stack_ch1;
        im_otherCh = im_stack_ch2;
    case 'Right'
        dropChIndex = 2;
        im_otherCh = im_stack_ch1;
        im_dropCh = im_stack_ch2;
    otherwise
        error('Unexpected value for ch_select')
end
end
function[paths2TifFiles,paths2Crop3DTifs,paths2MIPZTifs,paths2CropInfo,ID_str] = getPaths2Tifs()
mainDir = cd; % starting directory
fprintf('Please locate TIF file\n');
[~,d] = uigetfile([mainDir,'*.tif']);
mainDir = d;
paths2Crop3DDir = [mainDir,'3D Crops', filesep];
if ~exist(paths2Crop3DDir,'dir')
    mkdir(paths2Crop3DDir);
end
paths2MIPZDir = [mainDir,'MIPZ', filesep];
if ~exist(paths2MIPZDir,'dir')
    mkdir(paths2MIPZDir);
end
files = dir([mainDir,'*.tif']);
numOfTifFiles = numel(files);
paths2TifFiles = cell(numOfTifFiles,1);
paths2Crop3DTifs = cell(numOfTifFiles,1);
paths2MIPZTifs = cell(numOfTifFiles,1);
paths2CropInfo = cell(numOfTifFiles,1);
ID_str = cell(numOfTifFiles,1);
for n = 1:numOfTifFiles
    fname = files(n).name;
    path2Tif = [mainDir,fname];
    paths2TifFiles{n} = path2Tif;
    indx = regexp(path2Tif,filesep);
    ind1 = indx(end-1)+1;
    ind2 = indx(end)-1;
    ID_str{n} = [path2Tif(ind1:ind2),'_',fname(1:end-4)];
    CROP3D_fname = ['CROP3D_',ID_str{n},'.tif'];
    paths2Crop3DTifs{n} = [paths2Crop3DDir,CROP3D_fname];
    MIPZ_fname = ['MIPZ_',ID_str{n},'.tif'];
    paths2MIPZTifs{n} = [paths2MIPZDir,MIPZ_fname];
    cropInfo_fname = ['cropInfo_',ID_str{n},'.mat'];
    paths2CropInfo{n} = [paths2Crop3DDir,cropInfo_fname];
end
end

function[im_stack_ch1,im_stack_ch2,im_info] = getImagesAndInfo(path2TifFile,option1)
im_info = imfinfo(path2TifFile);
numOfSlices = numel(im_info);
Height = im_info.Height;
Width = im_info.Width;
numOfCh = 2;
Depth = numOfSlices/numOfCh;

switch option1
    case {'full','MIP_Z'}
        im_stack_ch1 = zeros(Height,Width,Depth);
        im_stack_ch2 = zeros(Height,Width,Depth);
        tic
        for n = 1:Depth
            indx_ch1 = 2*(n-1)+1;
            indx_ch2 = 2*n;
            im_stack_ch1(:,:,n) = imread(path2TifFile,indx_ch1);
            im_stack_ch2(:,:,n) = imread(path2TifFile,indx_ch2);
            elapsedTime = toc;
            if elapsedTime > 0.5
                fprintf('Loading slice %d of %d\n',n,Depth);
                tic
            end
        end
        fprintf('Finished %d of %d\n',Depth,Depth);
    case 'preview'
        n = 1;
        indx_ch1 = 2*(n-1)+1;
        indx_ch2 = 2*n;
        im_stack_ch1 = imread(path2TifFile,indx_ch1);
        im_stack_ch2= imread(path2TifFile,indx_ch2);
end
switch option1
    case 'MIP_Z'
        im_stack_ch1 = max(im_stack_ch1,[],3);
        im_stack_ch2 = max(im_stack_ch2,[],3);
end
 
end

function[a,b,xCenter,yCenter,orientation] = getEllipseFromImage(Im,thresholdValue)
if nargin == 2
    %thresholdFactor = 0.2;
    %thresholdValue = thresholdFactor*max(Im(:));
    V = double(Im > thresholdValue);
else
    thresholdValue = 0.25*max(Im(:));
    V = double(Im);
    V(Im < thresholdValue) = 0;
end

sumV = sum(V(:));
[numRows,numCols] = size(Im);
[X,Y] = meshgrid(1:numRows,1:numCols);
xCenter = sum(X(:).*V(:))/sumV;
yCenter = sum(Y(:).*V(:))/sumV;
XY = [X(:)-xCenter,Y(:)-yCenter].*sqrt([V(:),V(:)]); % intensity weighted centered coordinates
S = 1/(sumV)*(XY'*XY);
[vec,D] = eig(S);
orientation = atan(vec(2)/vec(1))*180/pi;
if orientation < 0
    orientation = orientation + 180;
end
a = 2*sqrt(D(1));
b = 2*sqrt(D(4));
xMin = uint16(max([1,xCenter - 1.5*max(a,b)]));
yMin = uint16(max([1,yCenter - 1.5*max(a,b)]));
xMax = uint16(min([numRows,xCenter + 1.5*max(a,b)]));
yMax = uint16(min([numCols,yCenter + 1.5*max(a,b)]));


figure(1)
hold off
imshow(Im(xMin:xMax,yMin:yMax),[]);
h = drawellipse('Center',[xCenter-xMin+1,yCenter-yMin+1],'SemiAxes',[a,b],'RotationAngle',orientation,'StripeColor','r');

end
