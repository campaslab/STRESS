classdef seg3D
    %SEG3D is a class containing methods for reconstructing surface
    %coordinates and analyzing surface curvatures of droplets starting from
    %tiff stacks. SEG3D can handle a single channel tiff z-stack as well as
    %a single channel tiff z-stack timelapse.
    %
    %   Created on September 5, 2019 by Elijah Shelton
    %   Last Modified February 22 2021 by Elijah
    %       * Supports radial analysis option
    %   Last Modified February 19 2021 by Elijah
    %       * Generates all directories for python analysis
    %       * Checks for python install, modules
    %       * Only plots surface points on first and last sequence
    %   Last Modified February 17 2021 by Elijah
    %       * Now checks for existence of destination folder 'Outputs'
    %       * Updated dialog for FILTER, FIDELITY, and ALPHA inputs
    %       * Removed some old warning messages
    %   Last Modified February 09 2021 by Elijah
    %       *fixed issues with timestamp changing for timeseries analysis
    %       *added functionality of specifying refactoring parameters
    %       *labeled axes and colorbars on final figures
    %   Last Modified February 05 2021 by Elijah
    %       *exports Input_Dict.mat file for use with subsequent analysis in python
    %       *exports just coordinate and curvature arrays as variables
    %   Modified October 21 2020 by Elijah
    %       *creates folder with tif name and timestamp, useful when
    %       different tifs are in the same directory*
    %   Modified September 14, 2020 by Elijah Shelton
    %   Modified July 30, 2020 by Elijah Shelton
    %   Modified April 25, 2020 by Elijah Shelton
    %   Fixed typo in getXYZfromVolume(); radiu -> radius; May 22, 2020 (E.S.)
    %
    %   This work expands upon code written for and ideas presented in the
    %   2017 Journal of Microscopy paper by Elijah Shelton, Friedhelm
    %   Serwane, & Otger Campas (https://doi.org/10.1111/jmi.12624).
    
    properties (Constant)
        DisplayOversampling = 1;
        PatchRadiusForMedianFilter = 0; % size of median filter applied to curvature measurements at the end
        PatchRadiusForIQRFilter = 0; % used to screen out coordinates during reconstruction using IQR filter applied to neighbor distances & errors in fitting parameters 
        modelShape = 'Ellipsoid'; % 'Sphere' or 'Ellipsoid'
        MinNumPerPatch = 18; % For F(x,y) poly2 fitting, N = 3 times num of fit params
        NumberOfIterations = 3; % number of iterations of patch fitting, testing sugests convergence at ~3
        SmallPatchLengthScale = 1; % recommend 1 or 2;
        FeatureKernelSize = 4; % This determines width of the kernel used to sample the feature channel
        SurfDens = 1; % Intended number of points per square pixel
        testing = 0; % Set to 1 if you would like to see graphic outputs from segmentation during run
    end
    properties
        alpha_percentile % Percentile excluded from upper and lower ends of distribution in python analysis
        ComputationTime % Time to run
        CoordErrors % Error in Coordinate Edge Fit
        Coordinates % Coordinates, in pixels (xy)
        CurrentIteration % Current iteration of patch fittings
        Curvatures  % Mean Curvatures, in inverse pixel dimensions (1/xy)
        Distances % Distances to nearest neighbors
        Ellipsoidals % structured variable containing information about ellipsoidal segmentation
        Features % Feature channel intensity from selected TIF
        FilterSize % size of gaussian filter applied to image
        FirstFrame
        fluorescence % 'Fluorescent Surface' or 'Fluorescent Interior'
        Indices % Indices of nearest neighbors
        IFT_milliNewtons_per_meter % Interfacial tension, in mN/m. Used in pythom analysis.
        k1 % 1st principal curvatures
        k2 % 2nd principal curvatures
        LastFrame
        minRp % Smallest patch radius permitted
        NumCoords % Number of coordinates
        Normals % Surface normals, calculated on last surface fitting.
        NS_kd % KD-tree model for neighbor searching
        PatchRadii % The radius of patch used in surface fitting
        Path2InputDictMat % Path to Input_Dict.mat file; Used to run python analysis
        Path2Mat  % Path to matlab results file
        Path2MatDir  % Path to directory containing matlab results file
        Path2Tif  % User specificied path to tiff stack containing drop image to analyze
        Path2TifDir  % Path to directory containing tiff stack
        Path2Feat % Path to tiff stack containing feature to analyze.
        proj2Fit % If 1 coordinates are projected onto fitted surfaces in the final interations.
        segmentMethod % 'Local Edge Fit' or 'Local Edge Fit - Parallelized'
        ReferenceCoordinates % For adding location of spatial references, used in Python analysis
        SurfaceArea
        TifName
        Timestamp_analysis % Tells you when the analysis was started.
        Timestep %
        vsx  % voxel size in x (um)
        vsz  % voxel size in z (um)
        deg_lbdv_fit % degree used for lebedev fit
        MAX_lbdv_fit_PTS; % 0 or 1; 1 instructs python analysis to use maximum spatial resolution possible. 
    end
    properties (Dependent)
    end
    methods (Static)
        function [] = notes()
            fprintf(...
                ['This version of seg3D requires the following add-ons:\n',...
                'Curve Fitting Toolbox\n',...
                'Image Processing Toolbox\n',...
                'Optimization Toolbox\n',...
                'Parallel Computing Toolbox\n',...
                'Statistic and Machine Learning Toolbox\n']...
                )
        end
        function [XYZ,ErrOut,FitParOut] = cleanUpXYZCoords(XYZ,boxLimits,ErrIn,FitParIn,patchRadius,edgeFit,parallelize)
            if nargin == 6
                parallelize = 0;
            end
            %% Check inside box
            xMin = boxLimits(1,1);
            xMax = boxLimits(2,1);
            yMin = boxLimits(1,2);
            yMax = boxLimits(2,2);
            zMin = boxLimits(1,3);
            zMax = boxLimits(2,3);
            xGood = (XYZ(:,1)>xMin) & (XYZ(:,1)<xMax);
            yGood = (XYZ(:,2)>yMin) & (XYZ(:,2)<yMax);
            zGood = (XYZ(:,3)>zMin) & (XYZ(:,3)<zMax);
            indInBox = xGood & yGood & zGood;
            XYZ(~indInBox,:) = nan;
            
            %% InterQuartileRange Filters
            [indPatch] = seg3D.getNeighborIndices( XYZ(:,1), XYZ(:,2), XYZ(:,3),patchRadius,parallelize);
            [numCoords,~] = size(XYZ);
            numNeighbors = nan(numCoords,1);
            for n = 1:numCoords
                numNeighbors(n) = length(indPatch{n});
            end
            
            
            [~,indIQR_err] = seg3D.interQuartileRangeFilter(log(ErrIn),'Upper'); % The order of magnitude of the errors is a better predictor
            [~,numFitPar] = size(FitParIn);
            indIQR_fPar = cell(numFitPar,1);
            for n=numFitPar
                [~,indIQR_fPar{n}] = seg3D.interQuartileRangeFilter(FitParIn(:,n),'Both');
            end
            [~,indIQR_nn] = seg3D.interQuartileRangeFilter(numNeighbors,'Lower');
            if edgeFit
                ind_accept = indIQR_err&indIQR_nn&indInBox;
            else
                ind_accept = true(size(indIQR_err));
            end    
            %ind_accept = ~isnan(XYZ(:,1));
            x = XYZ(ind_accept,1);
            y = XYZ(ind_accept,2);
            z = XYZ(ind_accept,3);
            ErrOut = ErrIn(ind_accept);
            close all
            scatter3(x,y,z,[],numNeighbors(ind_accept),'.');
            hold on
            x_reject = XYZ(~ind_accept,1);
            y_reject = XYZ(~ind_accept,2);
            z_reject = XYZ(~ind_accept,3);
            scatter3(x_reject, y_reject, z_reject,'.r');
            % Exclude Low density, high error coordinates;
            %warning('Need to fix error filtering...')
            indInclude = ind_accept;
            %% ORIGINAL
            ErrOut = ErrIn(indInclude);
            FitParOut = FitParIn(indInclude,:);
            x = XYZ(indInclude,1);
            y = XYZ(indInclude,2);
            z = XYZ(indInclude,3);
            center = nanmean(XYZ);
            xCntrd = x - center(1);
            yCntrd = y - center(2);
            zCntrd = z - center(3);
            [azi,elv,R] = cart2pol(xCntrd,yCntrd,zCntrd);
            % CONSIDER AN INTERPOLATED RESAMPLING STEP HERE...
            R_out = R;
%             N = length(x);
%             R_out = nan(N,1);
%             [indPatch] = seg3D.getNeighborIndices(xCntrd,yCntrd,zCntrd,patchRadius,parallelize);
%             for n=1:N
%                 indNbrs = indPatch{n};
%                 R_patch = R(indNbrs);
%                 R_out(n) = median(R_patch);
%                 R_out(n) = R(n);
%             end
            
            [xCntrd,yCntrd,zCntrd] = pol2cart(azi,elv,R_out);
            x = xCntrd + center(1);
            y = yCntrd + center(2);
            z = zCntrd + center(3);
            XYZ = [x,y,z];
        end
        function [H,k1,k2] = curvFromPoly2Fit(theta,Xq)
            %p00 = theta(1);
            p10 = theta(2);
            p01 = theta(3);
            p11 = theta(4);
            p20 = theta(5);
            p02 = theta(6);
            
            x = Xq(1);
            y = Xq(2);
            z = Xq(3);
            
            x0 = 1;
            x1 = x;
            x2 = y;
            %x3 = x.*y;
            %x4 = x.*x;
            %x5 = y.*y;
            
            X = [x0, x1, x2];
            
            tht_u = [p10; 2*p20; p11*y];
            hu = X*tht_u;
            tht_v = [p01; p11; 2*p02];
            hv = X*tht_v;
            tht_uu = [2*p20; 0; 0];
            huu = X*tht_uu;
            tht_vv = [2*p02; 0; 0];
            hvv = X*tht_vv;
            tht_uv = [p11; 0; 0];
            huv = X*tht_uv;
            
            %  Weisstein, Eric W. "Mean Curvature." From MathWorld--A Wolfram Web Resource. http://mathworld.wolfram.com/MeanCurvature.html 
            H = ((1+hv^2)*huu-2*hu*hv*huv+(1+hu^2)*hvv)./(2*(1+hu^2+hv^2)^(3/2));
            k1 = 2*max([p02,p20]); % Strictly speaking, this is the first principal curvature at the centroid. Will provide poor approx if query point is far from center/
            k2 = 2*min([p02,p20]); % Strictly speaking, this is the second principal curvature at the centroid. Will provide poor approx if query point is far from center
%             if H < 0
%                 warning('Mean curvature is negative...check orientation')
%             end
        end
        function [x,y] = ellipseCoords(a,b,orientation,N,center)
            if nargin < 5
                center = [0,0];
            end
            t = linspace(0,2*pi,N+1);
            t = t(1:end-1);
            % centered, level ellipse
            x = a*cos(t);
            y = b*sin(t);
            % centered rotated ellipse
            [x,y] = seg3D.rotateCoordsXYPlane(x,y,orientation);
            % translated, rotated ellipse
            x = x + center(1);
            y = y + center(2);
        end
        function [XYZ,rotMatrix] = ellipsoidCoords(semiAxesLengths,numCoords,center,rotMethod,rotParameters)
            % Generates a set of coordinates on the surface on an ellipsoid
            % defined by 'semiAxesLengths', 'eulerAngles' and 'center'
            %
            % DEFINE LOCAL COORDINATES
            z = linspace(1-1/numCoords,1/numCoords-1,numCoords);
            radius=sqrt(1-z.^2);
            goldenAngle = pi*(3-sqrt(5));
            theta = goldenAngle*(1:numCoords);
            XYZ_local = zeros(numCoords,3);
            XYZ_local(:,1) = semiAxesLengths(1)*radius.*cos(theta);
            XYZ_local(:,2) = semiAxesLengths(2)*radius.*sin(theta);
            XYZ_local(:,3) = semiAxesLengths(3)*z;
            
            % ROTATE COORDINATES
            switch rotMethod
                case 'Euler Angles'
                    eulerAngles = rotParameters;
                    c1 = cos(eulerAngles(1));
                    c2 = cos(eulerAngles(2));
                    c3 = cos(eulerAngles(3));
                    s1 = sin(eulerAngles(1));
                    s2 = sin(eulerAngles(2));
                    s3 = sin(eulerAngles(3));
                    % https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
                    rotMatrix = [    c2              -c3*s2               s2*s3;...
                        c1*s2    c1*c2*c3 - s1*s3   -c3*s1 - c1*c2*s3;...
                        s1*s2    c1*s3 + c2*c3*s1    c1*c3 - c2*s1*s3];
                case 'Rotation Matrix'
                    rotMatrix = rotParameters;
                otherwise
                    error('Unexpected value for ''rotMethod''.')
            end
            
            XYZ_rot = rotMatrix*XYZ_local';
            % TRANSLATE COORDINATES
            XYZ = XYZ_rot' + ones(numCoords,1)*center;
        end
        function [estCenter,estRadius] = estCenAndRad3D(V,X,Y,Z,fluorescence,testing)
            if nargin == 5
                testing = 0;
            end
            thresholdValue = 0.2*max(V(:));
            V_bool = double(V > thresholdValue);
            sumV = sum(V_bool(:));
            xCOM = sum(X(:).*V_bool(:))/sumV;
            yCOM = sum(Y(:).*V_bool(:))/sumV;
            zCOM = sum(Z(:).*V_bool(:))/sumV;
            
            XYZ = [X(:)-xCOM,Y(:)-yCOM,Z(:)-zCOM].*sqrt([V_bool(:),V_bool(:),V_bool(:)]); % intensity weighted centered coordinates
            
            S = 1/(sumV)*(XYZ'*XYZ);
            [rotMatrix,D] = eig(S);
            switch fluorescence
                case 'Fluorescent Interior'
                    R = sqrt(5*diag(D));
                case 'Fluorescent Surface'
                    R = sqrt(3*diag(D));
                otherwise
                    error('Unexpected value for ''fluorescence''')
            end
            
            a = R(1); % smallest semiaxis length
            b = R(2); % median semiaxis length
            c = R(3); % largest semiaxis length
            
            estCenter = [xCOM,yCOM,zCOM];
            estRadius = (a*b*c)^(1/3);
            
            MIP = sum(V,3);
            if testing
                %% CIRCLE CHECK
                %close all
                %imshow(MIP,[]);
                %hold on
                %plot(yCOM,xCOM,'.r')
                %N = 100;
                %[x_ell,y_ell] = seg3D.ellipseCoords(estRadius,estRadius,0,N,estCenter);
                %plot(y_ell,x_ell,'-r')
                %% ELLIPSOID CHECK
                N = 1000;
                XYZ_ell = seg3D.ellipsoidCoords(R,N,estCenter,'Rotation Matrix',rotMatrix);
                close all
                [~,~,P] = size(V);
                for p = 1:P
                    inPlane = (XYZ_ell(:,3) >= (p-1))&(XYZ_ell(:,3) < p);
                    XY = XYZ_ell(inPlane,1:2);
                    imshow(V(:,:,p),[]);
                    hold on
                    plot(XY(:,1),XY(:,2),'.r')
                    pause(0.1)
                end
                %                 %% SANDBOX
                %                 R0 = [5 7 10];
                %                 EA = [pi/4 pi/4 pi/4];
                %                 N = 1000;
                %                 C0 = [2 3 4];
                %                 [XYZ_ea,~] = seg3D.ellipsoidCoords(R0,N,C0,'Euler Angles',EA);
                %
                %                 C1 = mean(XYZ);
                %                 XYZ_c = XYZ - C1;
                %                 S = (1/N)*(XYZ_c'*XYZ_c);
                %                 [vec,D] = eig(S);
                %                 R1 = sqrt(3*diag(D));
                %
                %                 [XYZ_rm,~] = seg3D.ellipsoidCoords(R1,N,C1,'Rotation Matrix',vec);
                %                 close all
                %                 scatter3(XYZ_ea(:,1),XYZ_ea(:,2),XYZ_ea(:,3),'.k');
                %                 hold on
                %                 scatter3(XYZ_rm(:,1),XYZ_rm(:,2),XYZ_rm(:,3),'or');
                %                 axis equal
            end
        end
        function [estCenter,semiAxesLengths,rotMatrix] = estEllipsoidFromVolume(V,X,Y,Z,fluorescence,testing)
            if nargin == 5
                testing = 0;
            end
            thresholdValue = 0.2*max(V(:));
            V_bool = double(V > thresholdValue);
            sumV = sum(V_bool(:));
            xCOM = sum(X(:).*V_bool(:))/sumV;
            yCOM = sum(Y(:).*V_bool(:))/sumV;
            zCOM = sum(Z(:).*V_bool(:))/sumV;
            
            XYZ = [X(:)-xCOM,Y(:)-yCOM,Z(:)-zCOM].*sqrt([V_bool(:),V_bool(:),V_bool(:)]); % intensity weighted centered coordinates
            
            S = 1/(sumV)*(XYZ'*XYZ);
            [rotMatrix,D] = eig(S);
            switch fluorescence
                case 'Fluorescent Interior'
                    semiAxesLengths = sqrt(5*diag(D));
                case 'Fluorescent Surface'
                    semiAxesLengths = sqrt(3*diag(D));
                otherwise
                    error('Unexpected value for ''fluorescence''')
            end
            estCenter = [xCOM,yCOM,zCOM];
            
            if testing
                %% CIRCLE CHECK
                a = semiAxesLengths(1); % smallest semiaxis length
                b = semiAxesLengths(2); % median semiaxis length
                c = semiAxesLengths(3); % largest semiaxis length
                estRadius = (a*b*c)^(1/3);
                MIP = sum(V,3);
                close all
                imshow(MIP,[]);
                hold on
                plot(yCOM,xCOM,'.r')
                N = 100;
                [x_ell,y_ell] = seg3D.ellipseCoords(estRadius,estRadius,0,N,estCenter);
                plot(y_ell,x_ell,'-r')
                pause()
                %% ELLIPSOID CHECK
                N = 1000;
                XYZ_ell = seg3D.ellipsoidCoords(semiAxesLengths,N,estCenter,'Rotation Matrix',rotMatrix);
                close all
                [~,~,P] = size(V);
                for p = 1:P
                    inPlane = (XYZ_ell(:,3) >= (p-1))&(XYZ_ell(:,3) < p);
                    XY = XYZ_ell(inPlane,1:2);
                    imshow(V(:,:,p),[]);
                    hold on
                    plot(XY(:,1),XY(:,2),'.r')
                    pause(0.1)
                end
            end
        end
        function [estCenter,semiAxesLengths,rotMatrix] = estEllipsoidFromCoordinates(XYZ,parallelize)
            if nargin == 1
                parallelize = 0;
            end
            samplingDensity = seg3D.getSamplingDensity(XYZ,parallelize);
            indKeep = ~isnan(XYZ(:,1))&isfinite(XYZ(:,1));
            X = XYZ(indKeep,1);
            Y = XYZ(indKeep,2);
            Z = XYZ(indKeep,3);
            samplingDensity = samplingDensity(indKeep);
            W = 1./samplingDensity;
            sumW = sum(W);
            xCOM = sum(X.*W)/sumW;
            yCOM = sum(Y.*W)/sumW;
            zCOM = sum(Z.*W)/sumW;
            
            XYZ = [X(:)-xCOM,Y(:)-yCOM,Z(:)-zCOM].*sqrt([W,W,W]); % weighted centered coordinates
            
            S = 1/(sumW)*(XYZ'*XYZ);
            [rotMatrix,D] = eig(S);
            semiAxesLengths = sqrt(3*diag(D));
            estCenter = [xCOM,yCOM,zCOM];
        end
        function [startP,x_trimmed,y_trimmed] = estFitParameters(X,Y,fluorescence,testing)
            if nargin < 4
                testing = 0; % testing should be off be default
            end
            switch fluorescence
                case 'Fluorescent Interior'
                    % FIT MODEL
                    % h(x) = (amp+lambda*(x-mu))/(1 + exp(-k*(x-mu)) + offset
                    if length(X) < 5
                        warning('Too few points in trace (length(X) < 5)')
                        startP = nan(1,5);
                        x_trimmed = X;
                        y_trimmed = Y;
                        return
                    end
                    %% FIND X1
                    x1 = X(1);
                    y1 = Y(1);
                    X1 = [x1 y1];
                    %% FIND X3
                    dY = 0.5*diff([Y(1); Y]) + 0.5*diff([Y; Y(end)]);
                    [~,indX3] = max(abs(dY));
                    x3 = X(indX3);
                    y3 = Y(indX3);
                    X3 = [x3 y3];
                    %% FIND X5
                    x5 = X(end);
                    y5 = Y(end);
                    X5 = [x5 y5];
                    %% FIND X2 & X4
                    ddY = 0.5*diff([dY(1); dY]) + 0.5*diff([dY; dY(end)]);
                    magLddY = ddY;
                    magLddY(1) = nan;
                    magLddY(indX3:end) = nan;
                    [~, indX2] = max(magLddY);
                    x2 = X(indX2);
                    magRddY = ddY;
                    magRddY(end) = nan;
                    magRddY(1:indX3) = nan;
                    [~, indX4] = max(magRddY);
                    x4 = X(indX4);
                    fullWidth = x4 - x2;
                    halfWidth = fullWidth/2;
                    dX = X(2)-X(1);
                    indX2 = indX3 - round(halfWidth/dX); % REDEFINE indX2
                    if indX2 < 2 || indX2 > (length(X)-2)
                        indX2 = 2;
                    end
                    indX4 = indX3 + round(halfWidth/dX); % REDEFINE indX4
                    if indX4 < 3 || indX4 > (length(X)-1)
                        indX4 = length(X) - 1;
                    end
                    x2 = X(indX2);
                    y2 = Y(indX2);
                    x4 = X(indX4);
                    y4 = Y(indX4);
                    X2 = [x2 y2];
                    X4 = [x4 y4];
                    % ESTIMATE K
                    k = 4/(x4-x2);
                    if y4 > y2
                        k = -k;
                    end
                    width = abs(1/k);
                    % ESTIMATE CENTER
                    center = x3;
                    % ESTIMATE AMP
                    amp = 2*y3;
                    % ESTIMATE OFFSET & LAMBDA
                    if k > 0 % EXPECTED
                        offset = y5;
                        lambda = (y2 - y1)/(x2-x1);
                    else % UNUSUAL, BUT IT HAPPENS...
                        offset = y1;
                        lambda = (y5 - y4)/(x5-x4);
                    end
                    if ~isfinite(lambda)
                        error('lambda must be finite')
                    end
                    %% ESTIMATE STARTING FIT PARAMETERS
                    startP = [k center lambda amp offset];
                    %% TRIM COORDINATES
                    x_trimmed = X(indX2:indX4);
                    y_trimmed = Y(indX2:indX4);
                    if ~isfinite(sum(startP))
                        error('startP must contain 5 finite numbers')
                    end
                    if isnan(sum(startP))
                        error('startP must contain 5 finite numbers')
                    end
                    if length(startP)<5
                        error('startP must contain 5 finite numbers')
                    end
                    if testing
                        h = @(r) (r(4) + r(3)*(x_trimmed-r(2)))./(1+exp(r(1)*(x_trimmed-r(2)))) + r(5);
                        %h = @(r) (r(3))./(1+exp(-r(1)*(x_trimmed-r(2)))) + r(4);
                        yFit = h(startP);
                        close all
                        plot(X,Y,'ok')
                        hold on
                        plot(x_trimmed,yFit,'r')
                        plot([x1 x2 x3 x4 x5],[y1 y2 y3 y4 y5],'.');
                    end
                case 'Fluorescent Interior Old'
                    % FIT MODEL
                    % h(x) = amp*exp(-(x-mu)^2/lambda^2)/(1 + exp((x-mu)/width)) + offset
                    % Double check that...^ :-/
                    gradientThresholdFactor = 0.4; % Consider increasing to 0.4...
                    %warning('gradientThresholdFactor may be in need of adjustment...')
                    % ESTIMATE CENTER (mu)
                    dX = X(2) - X(1);
                    dY = diff(Y); % Calculate differentials at each segment;
                    [minDY, minDYind] = min(dY); % Determine minimum differential index
                    centerIndex = minDYind;
                    center = X(centerIndex); % Use minimum differential index to estimate center
                    % ESTIMATE EDGE WIDTH (width)
                    width1 =std((find(dY<gradientThresholdFactor*minDY)))*dX;
                    ddY = diff(dY);
                    %[minddY, ind1] = min(ddY);
                    ddY_rhs = ddY(minDYind:end);
                    [maxddY, ind_width] = max(ddY_rhs);
                    if isempty(ind_width)
                        warning('Bad Trace')
                        startP = nan(1,5);
                        x_trimmed = [];
                        y_trimmed = [];
                        return
                    end
                    ind1 = minDYind - ind_width;
                    
                    if isempty(ind1)
                        ind1 = 1;
                    elseif isnan(ind1) || ind1 < 1 || ind1 > length(X)
                        ind1 = 1;
                    end
                    width = ind_width*dX;
                    % TRIM COORDINATES
                    startTrim = floor(centerIndex - 3*width/dX);
                    if startTrim < 1 || startTrim > length(X) || isempty(startTrim) || isnan(startTrim)
                        startTrim = 1;
                    end
                    endTrim = ceil(centerIndex + 4*width/dX);
                    if endTrim < 1 || endTrim > length(X) || isempty(endTrim) || isnan(endTrim)
                        endTrim = length(X);
                    end
                    trimInd = startTrim:endTrim;
                    x_trimmed = X(trimInd);
                    y_trimmed = Y(trimInd);
                    % ESTIMATE OFFSET (offset)
                    offset = min(y_trimmed);
                    % ESTIMATE AMPLITUDE (amp)
                    
                    amp1 = Y(ind1+1)-offset;
                    % ESTIMATE INTERNAL SIGNAL ATTENUATION LENGTH
                    depth = X(ind1+1) - x_trimmed(1);
                    attenuation = amp1/y_trimmed(1);
                    lambda = depth/sqrt(log(attenuation));
                    if ~isfinite(lambda)||~isreal(lambda)
                        lambda = 1000*length(Y);
                    end
                    % ESTIMATED FIT PARAMETERS
                    startP = [width, center, lambda, amp1, offset];
                case 'Fluorescent Surface'
                    %startP: sigma mu amp offset of Gaussian
                    % ESTIMATE CENTER (mu)
                    if length(X) < 3
                        warning('X2 is too short; the ray is way too short.')
                        startP = NaN(1,4);
                        x_trimmed = NaN;
                        y_trimmed = NaN;
                        return
                    end
                    amp1=max(Y);
                    dX=(X(2)-X(1));
                    %calc mean, stdev of whole trace first guess of peak center and width
                    indCenter=round(mean(find(Y>0.4*amp1)));
                    indCenterStd=round(std(find(Y>0.1*amp1)));
                    if isempty(indCenter) ||isnan(indCenter)
                        startP=[nan nan nan nan];
                    else
                        center=X( indCenter);
                        centerStd=dX*indCenterStd;
                        %find coords with intens. values around the peak
                        indForCenter=find(abs(X-center)<1.5*centerStd);
                        indCenter=round(mean(find(Y(indForCenter)>0.4*amp1)));
                        if isnan(indCenter)
                            startP=[nan nan nan nan];
                        else
                            centerIndex = indForCenter(indCenter);
                            center=X( centerIndex );
                            width=std((find(Y>0.2*amp1)))*dX;
                            offset=0;
                            startP=[width center amp1 offset];
                        end
                    end
                    x_trimmed = X;
                    y_trimmed = Y;
                    %error('NEED TO ESTIMATE FIT PARAMETERS')
                otherwise
                    error('Unexpected value for ''fluorescence''.')
            end
            if testing
                % FLUORESCENT INTERIOR
                h = @(r) (r(4) + r(3)*(x_trimmed-x_trimmed(1)))./(1+exp(r(1)*(x_trimmed-r(2)))) + r(5);
                %h = @(r) (r(3))./(1+exp(-r(1)*(x_trimmed-r(2)))) + r(4);
                yFit = h(startP);
                close all
                plot(X,Y,'ok')
                hold on
                plot(x_trimmed,yFit,'r')
                pause()
            end
        end
        function [path2CoordsAndCurves] = exportCoordsAndCurves(path2MatFile,pointCloudArray,meanCurvatureArray,data,timestamp)
            pathDissassembled = regexp(path2MatFile,filesep,'split');
            matFilename = pathDissassembled{end};
            path2MatDir = strjoin(pathDissassembled(1:end-1),filesep);
            CoordsAndCurvesSubFolder = ['CoordsAndCurves',timestamp];
            path2CoordsAndCurvesDir = [path2MatDir,filesep,CoordsAndCurvesSubFolder];
            if ~exist(path2CoordsAndCurvesDir,'dir')
                mkdir(path2CoordsAndCurvesDir);
            end
            fileSuffix =  [matFilename(3:6),'.mat'];
            CoordsAndCurvesMatFilename = ['justCoordinatesAndCurvatures',fileSuffix];
            path2CoordsAndCurves = [path2CoordsAndCurvesDir,filesep,CoordsAndCurvesMatFilename];
            save(path2CoordsAndCurves,'meanCurvatureArray','pointCloudArray','data');
        end
        function [area] = findSurfaceArea(XYZ,displaySurface)
            if nargin == 1
                displaySurface = 0;
            end
            % To calculate surface area, we can represent the surface as a
            % mesh of rectangles. The total surface area is calculated by
            % summing the area of each of the rectangles.
            k = floor(log((length(XYZ))^(1/2))/log(2));
            n = 2^k-1;
            [x,y,z,R] = seg3D.meshPointCloud(XYZ,n);
            if displaySurface
                close all
                s = surf(x,y,z,R);
                s.EdgeColor = 'none';
                s.FaceLighting = 'gouraud';
                axis square
            end
            [m,n] = size(z);
            area = 0;
            for i = 1:m-1
                for j = 1:n-1
                    v0 = [x(i,j)     y(i,j)     z(i,j)    ];
                    v1 = [x(i,j+1)   y(i,j+1)   z(i,j+1)  ];
                    v2 = [x(i+1,j)   y(i+1,j)   z(i+1,j)  ];
                    v3 = [x(i+1,j+1) y(i+1,j+1) z(i+1,j+1)];
                    a = v1 - v0;
                    b = v2 - v0;
                    c = v3 - v0;
                    A = 1/2*(norm(cross(a, c)) + norm(cross(b, c)));
                    area = area + sum(A);
                end
            end
            
        end
        function [fitPar,yFit,MSE,FVU] = fitBoundary(x,y,startPar,fluorescence)
            % optimoptions and lsqnonlin are relatively slow
            options = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt','display','off','MaxFunctionEvaluations',1500);
            switch fluorescence
                case 'Fluorescent Interior'
                    % r = [k mu amp offset]
                    costFun = @(r) (r(4)+r(3)*(x-r(2)))./(1+exp(r(1)*(x-r(2)))) + r(5) - y;
                    h = @(r) (r(4)+r(3)*(x-r(2)))./(1+exp(r(1)*(x-r(2)))) + r(5);
                case 'Fluorescent Interior Old'
                    % r = [sigma mu lambda amp offset]
                    costFun = @(r) (r(4)*exp(-(x-r(2)).*(x-r(2))./(r(3)*r(3))))./(1+exp((x-r(2))./r(1))) + r(5) - y;
                    h = @(r) (r(4)*exp(-(x-r(2)).*(x-r(2))./(r(3)*r(3))))./(1+exp((x-r(2))./r(1))) + r(5);
                case 'Fluorescent Surface'
                    % r = [sigma mu amp offset];
                    costFun = @(r) (r(3)*exp(-(x-r(2)).*(x-r(2))./(r(1)*r(1)))) + r(4) - y;
                    h = @(r) (r(3)*exp(-(x-r(2)).*(x-r(2))./(r(1)*r(1)))) + r(4);
                otherwise
                    error('Unexpected value for ''fluorescence''.')
            end
            try 
                fitPar = lsqnonlin(costFun,startPar,[],[], options);
            catch
                warning('You got problems...')
                fitPar = nan(size(startPar));
            end
            if ~isfinite(sum(fitPar))
                warning('Returning non-finite fit parameters. Model is poorly conditioned for data.')
                fitPar = nan(size(startPar));
            end
            yFit = h(fitPar);
            J = (yFit - y);
            %rmsErr = sqrt(J'*J/length(J));
            MSE = mean(J.^2); % Mean Square Error;
            FVU = MSE/var(y); % Fraction Of Variance Unexplained (by fit)
        end
        function [theta,J,h] = fitPoly2(x,y,z)
            %% fitPoly2
            % INPUTS: x,y,z are all column vectors
            % OUTPUTS: theta: fitting parameters, J: cost function
            m = length(x);
            
            x0 = ones(m,1);
            x1 = x;
            x2 = y;
            x3 = x.*y;
            x4 = x.*x;
            x5 = y.*y;
            
            X = [x0, x1, x2, x3, x4, x5];
            Y = z;
            
            A = X'*X;
            theta = pinv(A)*X'*Y;
            h = X*theta;
            R2 = (h-Y)'*(h-Y);
            J = (1/(2*m))*R2;
            
            %% TEST FIT
            % close all
            % scatter3(x,y,z,'.');
            % hold on
            % scatter3(x,y,h,'o');
        end
        function [XYZ,Err,FitParams] = fitTraces(traceStarts,traceEnds,F,fluorescence,edgeFit,testing)
            if nargin == 5
                testing = 0; % Change to 1 to display plots
            end
            N = length(traceEnds);
            XYZ = nan(N,3);
            Err = nan(N,1);
            if edgeFit
                switch fluorescence
                    case 'Fluorescent Interior'
                        numFitPar = 5;
                    case 'Fluorescent Surface'
                        numFitPar = 4;
                    otherwise
                        error('Unexpected value for ''fluorescence''.')
                end
            else
                numFitPar = 1;
            end
            FitParams = nan(N,numFitPar);
            samplingLength = 1;
            % This loop is relatively slow. Consider parallelizing.
            for n = 1:N
                rStart = traceStarts(n,:);
                rEnd = traceEnds(n,:);
                [traceCoords,sampVec] = ...
                    seg3D.generateLineCoords(rStart, rEnd, samplingLength);
                [t,Int] = seg3D.getTraceToFit(traceCoords,F);
                if edgeFit
                    [startP,t_trimmed,Int_trimmed] = seg3D.estFitParameters(t,Int,fluorescence,testing);
                    [fitPar,yFit,~,FVU] = seg3D.fitBoundary(t_trimmed,Int_trimmed,startP,fluorescence);
                    XYZ(n,:) = fitPar(2)*sampVec + rStart;
                    Err(n) = FVU;
                    FitParams(n,:) = fitPar;
                else
                    switch fluorescence
                        case 'Fluorescent Interior'
                            dInt = abs(diff(Int));
                            dt = (t(2)-t(1));
                            fitPar = t(dInt==max(dInt))+0.5*dt;
                        case 'Fluorescent Surface'
                            fitPar = t(Int==max(Int));
                        otherwise
                            error('Unexpected value for ''fluorescence''.')
                    end
                    XYZ(n,:) = fitPar*sampVec + rStart;
                    Err(n) = 0;
                    FitParams(n,:) = fitPar;
                end
                if testing
                    close all
                    % MAY ONLY WORK FOR FLUORESCENT INTERIOR
                    plot(t_trimmed,Int_trimmed,'-b')
                    hold on
                    plot(t_trimmed,yFit,'-r')
                    plot(fitPar(2),max(yFit),'o')
                    xEdge = [1 1]*fitPar(2);
                    yEdge = [0 1]*fitPar(4);
                    plot(xEdge,yEdge,'-k')
                    pause()
                end
                %close all
                %plot(XY(:,1),XY(:,2),'.')
            end
        end
        function [path2InputDictMatFile] = generateInputDict(fullpath,INPUT_DICT)
            % make Outputs directory, if it doesn't exist
            reqDirs = {'Outputs','Pickled_LBDV_Files','Pickled_Data','Pickled_Sphere_Triangulations'};
            for n = 1:numel(reqDirs)
                if~exist(reqDirs{n},'dir')
                    mkdir(reqDirs{n});
                end
            end
            foldersInPath = regexp(fullpath,filesep,'split');

            SubFolder_We_Load_From = foldersInPath{end-1};
            MatFile_filepath = strjoin(foldersInPath(1:end-1),filesep);
            Tension_gamma_val = INPUT_DICT.IFT;
            Pixel_Size_Microns = INPUT_DICT.Pixel_size_microns;
            Use_PyCompadre = 0;
            Plot_Vtu_Outputs = 1;
            Output_Plot_Subfolder_Name = [ SubFolder_We_Load_From,'_OUTPUT'];
            deg_lbdv_fit = INPUT_DICT.deg_lbdv_fit;
            MAX_lbdv_fit_PTS = INPUT_DICT.MAX_lbdv_fit_PTS;
            deg_lbdv_Vol_Int = INPUT_DICT.deg_lbdv_Vol_Int;
            MAX_lbdv_vol_PTS = INPUT_DICT.MAX_lbdv_vol_PTS;
            deg_fit_Ellipsoid_Deviation_Analysis = INPUT_DICT.deg_fit_Ellipsoid_Deviation_Analysis;
            alpha_percentile_excl_AnisStress = INPUT_DICT.alpha_percentile_excl_AnisStress;
            alpha_percentile_excl_AnisCellStress = INPUT_DICT.alpha_percentile_excl_AnisCellStress;
            alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin = INPUT_DICT.alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin;
            Corr_Start_Dist = 0;
            drop_center_XYZ_um = INPUT_DICT.drop_center_XYZ_um;
            EK_center_XYZ_um = INPUT_DICT.EK_center_XYZ_um;
            if ~isempty(EK_center_XYZ_um) && ~isempty(drop_center_XYZ_um)
                Neha_Radial_analysis = true;
            else
                Neha_Radial_analysis = false;
            end
            
            Calc_Spat_Temp_Corrs = 0;
            smoothBSpline = 1;
            path2InputDictMatFile = ['Input_Dict.mat'];
            save(path2InputDictMatFile,...
                'SubFolder_We_Load_From',...
                'MatFile_filepath',...
                'Tension_gamma_val',...
                'Use_PyCompadre',...
                'Plot_Vtu_Outputs',...
                'Pixel_Size_Microns',...
                'Output_Plot_Subfolder_Name',...
                'deg_lbdv_fit',...
                'MAX_lbdv_fit_PTS',...
                'deg_lbdv_Vol_Int',...
                'MAX_lbdv_vol_PTS',...
                'deg_fit_Ellipsoid_Deviation_Analysis',...
                'alpha_percentile_excl_AnisStress',...
                'alpha_percentile_excl_AnisCellStress',...
                'alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin',...
                'Corr_Start_Dist',...
                'Neha_Radial_analysis',...
                'drop_center_XYZ_um',...
                'EK_center_XYZ_um',...
                'Calc_Spat_Temp_Corrs',...
                'smoothBSpline')
        end
        function [traceCoordinates,sampVec] = generateLineCoords(rStart,rV, samplingLength)
            dispVec = rV-rStart;
            lengthTrace = norm(dispVec);
            NPoints = floor(lengthTrace/samplingLength) + 1;
            sampVec = samplingLength*dispVec./norm(dispVec);
            traceCoordinates = ones(NPoints,1)*rStart + (0:(NPoints-1))'*sampVec;
        end
        function [Indices,NumNearestNeighbors] = getNeighborIndices(x,y,z,patchRadius,parallelize)
            if nargin == 4
                parallelize = 0;
            end
            N = length(x);
            patchRadii = patchRadius*ones(N,1);
            queryPoints = [x,y,z];
            Indices = cell(N,1);
            NumNearestNeighbors = nan(N,1);
            NS_kd = KDTreeSearcher(queryPoints);
            if parallelize
                parfor n = 1:N
                    [idx]=rangesearch(NS_kd,queryPoints(n,:),patchRadii(n));
                    Indices{n} = idx{1};
                    NumNearestNeighbors(n) = length(idx{1});
                end
            else
                for n = 1:N
                    [idx]=rangesearch(NS_kd,queryPoints(n,:),patchRadii(n));
                    Indices{n} = idx{1};
                    NumNearestNeighbors(n) = length(idx{1});
                end
            end
        end
        function [startRefined,endRefined] = getNewTraceVectors3D(XYZ,samplingLength,traceLength,parallelize)
            % Resample Surface
            [XYZ_out] = seg3D.ResampleSurface(XYZ,samplingLength);
            xNew = XYZ_out(:,1);
            yNew = XYZ_out(:,2);
            zNew = XYZ_out(:,3);
            
            % Calculate new centers after interpolating for even sampling
            xNew0 = mean(xNew);
            yNew0 = mean(yNew);
            zNew0 = mean(zNew);
            
            xNewCntrd = xNew - xNew0;
            yNewCntrd = yNew - yNew0;
            zNewCntrd = zNew - zNew0;
            
            % Calculate radial coordinates with newly centered coords.
            %[~,~,R] = cart2sph(xNewCntrd,yNewCntrd,zNewCntrd);
            patchRadius = 2;
            normVec = seg3D.getLocalNormals3D(xNew,yNew,zNew,patchRadius,parallelize);
            
            %% CHECK BELOW...
            R = traceLength;
            startRefined = [xNew,yNew,zNew] - normVec.*R/2; % Consider using fit parameters to size trace vectors...
            endRefined = [xNew,yNew,zNew] + normVec.*R/2;
            
            %VISUAL CHECK IF DESIRED...
            %close all
            %scatter3(startRefined(:,1),startRefined(:,2),startRefined(:,3),'.')
            %axis equal
            %hold on
            %scatter3(endRefined(:,1),endRefined(:,2),endRefined(:,3),'.')
            
        end
        function [x,y,z] = getNormSpiralCoords(N)
            z = linspace(1-1/N,1/N-1,N)';
            radius=sqrt(1-z.^2);
            goldenAngle = pi*(3-sqrt(5));
            theta = goldenAngle*(1:N)';
            x = radius.*cos(theta);
            y = radius.*sin(theta);
        end
        function [normVec] = getLocalNormals3D(x,y,z,patchRadius,parallelize)
            if nargin == 4
                parallelize = 0;
            end
            tic
            fprintf('       Finding normals for retrace...')
            N = length(x); % Number of coordinates
            normVec = nan(N,3);
            indPatch = seg3D.getNeighborIndices(x,y,z,patchRadius,parallelize);
            Xct = mean([x,y,z]);
            patchCoords = cell(N,1);
            for n = 1:N
                xPatch = x(indPatch{n});
                yPatch = y(indPatch{n});
                zPatch = z(indPatch{n});
                patchCoords{n} = [xPatch,yPatch,zPatch];
            end
            for n = 1:N
                xPatch = patchCoords{n}(:,1);
                yPatch = patchCoords{n}(:,2);
                zPatch = patchCoords{n}(:,3);
                Xq = [x(n),y(n),z(n)];
                X = [xPatch-nanmean(xPatch),yPatch-nanmean(yPatch),zPatch-nanmean(zPatch)];
                
                numInPatch = length(X);
                
                S = 1/(numInPatch-1)*(X'*X);
                [V,~] = eig(S);
                OrientMatrix = [V(:,3),V(:,2),V(:,1)]; % reverse vector order so surface normal is in z
                Xlq = Xq*OrientMatrix;
                Xlct = Xct*OrientMatrix;
                %Check that vector is pointing outward
                if (Xlct(3) - Xlq(3)) > 0
                    FlipUpsideDown = [1 0 0; 0 -1 0; 0 0 -1]; % Rotates 180 deg about x-axis
                    OrientMatrix = OrientMatrix*FlipUpsideDown;
                end
                nV = OrientMatrix(:,3)';
                normVec(n,:) = nV;
            end
            tNRT = toc;
            fprintf('(%3.1f seconds)\n',tNRT)
        end
        function [samplingDensity] = getSamplingDensity(XYZ,parallelize)
            if nargin == 1
                parallelize = 0;
            end
            X = XYZ(:,1);
            Y = XYZ(:,2);
            Z = XYZ(:,3);
            patchRadius = 2;
            [~,numNearestNeighbors] = seg3D.getNeighborIndices(X,Y,Z,patchRadius,parallelize);
            samplingDensity = (numNearestNeighbors)./(pi*patchRadius.^2);
        end
        function [t,I] = getTraceToFit(traceCoords,F)
            [numPoints,~] = size(traceCoords);
            t = (0:numPoints-1)';
            I = F(traceCoords);
            ind = ~isnan(I);
            I = I(ind);
            t = t(ind);
        end
        function [startTrace,endTrace] = getTraceVectorsEllipsoid(semiAxesLengths,N,startXYZ,rotMatrix)
            if nargin == 3
                rotMatrix = [1,0,0; 0 1 0; 0 0 1];
            end
            startTrace = (startXYZ'*ones(1,N))';
            rotMethod = 'Rotation Matrix';
            center = startXYZ;
            numCoords = N;
            [endTrace,~] = seg3D.ellipsoidCoords(2*semiAxesLengths,numCoords,center,rotMethod,rotMatrix);
        end
        function [XYZ,Err] = getXYZfromVolume(F_int,boxLimits,modelShape,modelParams,sampLen,fluorescence,patchRadius,parallelize,edgeFit,testing)
            N = 2^8; %% number of coarse coordinates
            if nargin == 9
                testing = 0;
            end
            switch modelShape
                case 'Sphere'
                    center = modelParams.center;
                    radius = modelParams.radius;
                    semiAxesLengths = [radius,radius,radius];
                    rotMatrix = [1 0 0; 0 1 0; 0 0 1];
                    %traceLength = 6*radius*ones(3,1);
                    startXYZ = center;
                    %[startTrace,endTrace] = seg3D.getTraceVectors(traceLength,N,startXYZ);
                case 'Ellipsoid'
                    center = modelParams.center;
                    semiAxesLengths = modelParams.semiAxesLengths;
                    rotMatrix = modelParams.rotMatrix;
                    startXYZ = center;
                otherwise
                    error('Unexpected value for ''modelShape''.')
            end
            fprintf('     Coarse graining surface (%d points)...\n',N)
            tic
            fprintf('       Determining trace vectors...')
            [startTrace,endTrace] = seg3D.getTraceVectorsEllipsoid(semiAxesLengths,N,startXYZ,rotMatrix);
            tGTV = toc;
            fprintf('(%3.1f seconds)\n',tGTV)
            tic
            fprintf('       Tracing and fitting intensities...')
            
            if parallelize
                [XYZ,Err,FitPar] = seg3D.parFitTraces(startTrace,endTrace,F_int,fluorescence,edgeFit,testing);
            else
                [XYZ,Err,FitPar] = seg3D.fitTraces(startTrace,endTrace,F_int,fluorescence,edgeFit,testing);
            end
            tFT = toc;
            fprintf('(%3.1f seconds)\n',tFT)
            tic
            fprintf('       Cleaning up coordinates...')
            [XYZ,Err,FitPar] = seg3D.cleanUpXYZCoords(XYZ,boxLimits,Err,FitPar,patchRadius,edgeFit,parallelize);
            tSM= toc;
            fprintf('(%3.1f seconds)\n',tSM)
            %% refine coordinates
            numRefinements = 2;
            for i = 1:numRefinements
                fprintf('     Refining point cloud (%d of %d)...\n',i,numRefinements)
                if edgeFit
                    medFPar1 = nanmedian(FitPar(:,1));
                    switch fluorescence
                        case 'Fluorescent Surface'
                            traceLength = 4*medFPar1;
                        case 'Fluorescent Interior'
                            traceLength = 16/medFPar1;
                        otherwise
                            error('Unexpected value for ''fluorescence''.')
                    end
                else
                    traceLength = nanmedian(FitPar(:,1));
                end
                % why is traceLength needed?
                [startRefined,endRefined] = seg3D.getNewTraceVectors3D(XYZ,sampLen,traceLength,parallelize);
                tic
                if i == 1
                    tFFT = (length(startRefined)/N)*tFT;
                end
                fprintf('       Tracing and fitting intensities...\n')
                
                fprintf('       Estimating %4.0f seconds...',tFFT)
                if parallelize
                    [XYZ,Err,FitPar] = seg3D.parFitTraces(startRefined,endRefined,F_int,fluorescence,edgeFit,testing);
                else
                    [XYZ,Err,FitPar] = seg3D.fitTraces(startRefined,endRefined,F_int,fluorescence,edgeFit,testing);
                end
                %[XYZ,Err,FitPar] = seg3D.crop2Box(XYZ,Err,FitPar,XYZLimits);
                fprintf('       Cleaning up coordinates...')
                [XYZ,Err,FitPar] = seg3D.cleanUpXYZCoords(XYZ,boxLimits,Err,FitPar,patchRadius,edgeFit,parallelize);
                tFFT = toc;
                fprintf('(%4.1f seconds)\n',tFFT);
                
            end
        end
        function [Vout, ind] = interQuartileRangeFilter(Vin,filterOut)
            scale = 1.5;
            Q1 = quantile(Vin,0.25);
            Q3 = quantile(Vin,0.75);
            IQR = Q3 - Q1;
            if nargin == 1
                filterOut = 'Both';
            end
            switch filterOut
                case 'Upper'
                    ind = (Vin <= (Q3 + scale*IQR));
                case 'Lower'
                    ind = (Vin >= (Q1 - scale*IQR));
                case 'Both'
                    ind = (Vin >= (Q1 - scale*IQR)) & (Vin <= (Q3 + scale*IQR));
                otherwise
                    error('Unexpected value for ''filterOut''.')
            end
            Vout = Vin(ind);
        end
        function [TimelapseObj] = makeTimelapseObj(featureOption)
            % Determine path to tif stack of drop
            fprintf('\nPlease locate the drop tif file you would like to analyze.\n');
            [f,path2TifDir] = uigetfile('*.tif');
            fprintf('Selected file:\n    %s\n', f)
            path2Tif = [path2TifDir,f];
            % Determine path to tif stack of feature
            if nargin == 0
                featureOption = 0;
            end
            if featureOption
                rsp = questdlg('Would you like to perform a FEATURE analysis?','Feature analysis','Yes','No','Exit','No');
                switch rsp
                    case 'Yes'
                        fprintf('\nPlease locate the feature tif file you would like to analyze.\n');
                        [f,d] = uigetfile('*.tif');
                        path2Feat = [d,f];
                    case 'No'
                        path2Feat = [];
                    case 'Exit'
                        TimelapseObj = cell(0);
                        return
                    otherwise
                        warning('Unexpected response. Quite unexpected. Rude!')
                        TimelapseObj = cell(0);
                        return
                end
            else
                path2Feat = [];
            end
            % Determine number of timesteps
            numOfTimesteps = input('How many timesteps? \n');
            % Determine the pixel sizes
            vsx = input('\nPlease enter the voxel size in x (microns) :  ');
            vsz = input('\nPlease enter the voxel size in z (microns) :  ');
            % Determine drop fluorecence labeling scheme
            dropFluorQuest = 'How is the drop labeled in your image?';
            fluInt = 'Fluorescent Interior';
            fluSurf = 'Fluorescent Surface';
            defFlu = fluInt;
            fluorescence = questdlg(dropFluorQuest,'Label Options',fluInt,fluSurf,defFlu);
            % Specify Interfacial Tension (optional)
            IFT = input('\nPlease enter the Interfacial Tension (mN/m) :  ');
            % Determine the segment methods
            % LOCAL EDGE FIT or QUICK EDGE DETECTION
            segMethQuest = 'Which segmentation method would you like to use?';
            lef = 'Local Edge Fit';
            qed = 'Quick Edge Detection';
            edgeMethod = questdlg(segMethQuest,'Segmentation Options',lef,qed,lef);
            % PARALLELIZE or SERIALIZE
            parQuest = 'Would you like to parallelize computation?';
            parMethod =  questdlg(parQuest,'Segmentation Options','Parallelize','Serialize','Parallelize');
            switch parMethod
                case 'Parallelize'
                    segmentMethod = [edgeMethod,' - Parallelized'];
                otherwise
                    segmentMethod = edgeMethod;
            end
            % Set image filter size
            filterPromptString = ['\nWe need to apply a 3D gaussian filter to the image.\n'...
                'What filter size do you want to use (in pixels)?\n',...
                '(Recommended: 1-10; preferred size is 1)\n'];
            FilterSize = input(filterPromptString);
            % Specify refaactoring parameters
            fprintf(['\nQUICK INFO ON FIDELITY SETTINGS:\n',...
                'Fidelity settings determine how many harmonic modes\n',...
                'to use in the harmonic representation of the surface.\n',...
                'Higher fidelity means more modes and an increased\n',...
                'ability to capture fine varaitions in shape, but it\n',...
                'also means a longer processing time. We recommend you\n',...
                'run it first at a lower fidelity, then re-run at a\n',...
                'higher level after other settings are dialed in.\n'])
            refactorResQuest = {'Set FIDELITY',...
                '',...
                'See Command Window for details.'};
            listString = {'Bottom of the barrel','Super low','Low','Medium','High','Super high'};
            refactorResLevel = listdlg('PromptString',refactorResQuest,...
                'ListString',listString,'SelectionMode','single');
            switch listString{refactorResLevel}
                case 'Super high'
                    deg_lbdv_fit = 20;
                case 'High'
                    deg_lbdv_fit = 17;
                case 'Medium'
                    deg_lbdv_fit = 14;
                case 'Low'
                    deg_lbdv_fit = 11;
                case 'Super low'
                    deg_lbdv_fit = 8;
                case 'Bottom of the barrel'
                    deg_lbdv_fit = 5;
                otherwise
            end
            
            MAX_lbdv_response = questdlg('Would you like to MAXIMIZE number of points used for Lebedev resampling?',...
                'MAX LBDV Points?','Yes','No','No');
            switch MAX_lbdv_response
                case 'Yes'
                    MAX_lbdv_response = questdlg('Are you sure? MAXIZING is VERY time consuming...',...
                        'MAX LBDV Points?','Let''s do it, I''m in no hurry.','No, nevermind.','No, nevermind');
                    switch MAX_lbdv_response
                        case 'Let''s do it, I''m in no hurry.'
                            MAX_lbdv_fit_PTS = 1;
                        case 'No, nevermind.'
                            MAX_lbdv_fit_PTS = 0;
                    end
                case 'No'
                    MAX_lbdv_fit_PTS = 0;
            end
            fprintf(['\nALPHA determines the percentile at which extreme values\n',...
                'are excluded when computing stress amplitudes.\n'])
            alpha_percentile = 0.05;
            fprintf('Default alpha value %1.2f reccommended.\n',alpha_percentile)
            QDString = sprintf('Proceed with alpha = %1.2f?',alpha_percentile);
            default_alpha = questdlg(QDString,'ALPHA SETTING','Yes','No, select custom value','Yes');
            switch default_alpha
                case 'Yes'
                    fprintf('Default alpha value %1.4f in use.\n',alpha_percentile)
                case 'No, select custom value'
                    while true
                        alpha_percentile = input('Please specify alpha percentile between 0 and 0.5:\n');
                        if alpha_percentile < 0
                            fprintf('ALPHA must be greater than 0\n')
                        elseif alpha_percentile > 0.5
                            fprintf('ALPHA must be less than 0.5\n')
                        else
                            fprintf('Custom alpha value %1.4f in use.\n',alpha_percentile)
                            break
                        end
                    end
            end
            
            % Do you have a reference coordinates to add?
            refCoordRsp = questdlg('Do you have a reference coordinate to add?','Reference coordinate','Yes','No','Yes');
            switch refCoordRsp
                case 'Yes'
                    fprintf(['\nIs the pixel size and z-step in your reference image\n',...
                        'the same as in the cropped stack of the droplet?\n'])
                    refPxSame = input('(y/n)\n>> ','s');
                    switch refPxSame
                        case 'y'
                            ref_vsx = vsx;
                            ref_vsz = vsz;
                        case 'n'
                            fprintf('Please specify pixel size (microns) in reference image.\n')
                            ref_vsx = input('>> ');
                            fprintf('Please specify z-step (microns) in reference image.\n')
                            ref_vsz = input('>> ');
                    end
                    fprintf('\nPlease specify the row, column, and page of the droplet center.\n');
                    RC.drop_row = input('\nROW: ');
                    RC.drop_col = input('\nCOLUMN: ');
                    RC.drop_page = input('\nPAGE (a.k.a. frame number in Z): ');
                    fprintf('Please specify the row, column, and page of the reference coordinate.');
                    RC.EK_row = input('\nROW: ');
                    RC.EK_col = input('\nCOLUMN: ');
                    RC.EK_page = input('\nPAGE (a.k.a. frame number in Z): ');
                    RC.drop_center_XYZ_um = [RC.drop_row*ref_vsx, RC.drop_col*ref_vsx RC.drop_page*ref_vsz];
                    RC.EK_center_XYZ_um = [RC.EK_row*ref_vsx, RC.EK_col*ref_vsx RC.EK_page*ref_vsz];
                    
                    fprintf('\nPlease specify unique sequence ID string.\n')
                    RC.sequence_ID_string = input('>> ','s');
                    
                    fprintf('\nPlease specify relative timestamp in hours.\n')
                    RC.timestamp_hrs = input('>> ','s');
                    
                case 'No'
                    RC.drop_row = [];
                    RC.drop_col = [];
                    RC.drop_page = [];
                    RC.EK_row = [];
                    RC.EK_col = [];
                    RC.EK_page = [];
                    RC.drop_center_XYZ_um = [];
                    RC.EK_center_XYZ_um = [];
                    RC.sequence_ID_string = [];
                    RC.timestamp_hrs = [];
                otherwise
                    error('Unexpected response.')
            end
            
            
            % Determine which frames to load for each timestep
            tifInfo = imfinfo(path2Tif);
            numOfFrames = length(tifInfo);
            numOfZSteps = numOfFrames/numOfTimesteps;
            % Get Timestamp
            TS = clock();
            timestamp = sprintf('_%d',TS(1:5));
            TimelapseObj = cell(numOfTimesteps,1);
            for t = 1:numOfTimesteps
                %% INFO FOR EACH TIMESTEP
                TimelapseObj{t} = seg3D();
                TimelapseObj{t}.Path2Tif = path2Tif; % path to tif file of droplet
                TimelapseObj{t}.Path2Feat = path2Feat; % path to tif file of feature to be analysed, i.e. cell membrane label
                TimelapseObj{t}.FirstFrame = (t-1)*numOfZSteps + 1;
                TimelapseObj{t}.LastFrame = t*numOfZSteps;
                TimelapseObj{t}.Timestep = t;
                TimelapseObj{t}.vsx = vsx;
                TimelapseObj{t}.vsz = vsz;
                TimelapseObj{t}.IFT_milliNewtons_per_meter = IFT;
                TimelapseObj{t}.fluorescence = fluorescence;
                TimelapseObj{t}.segmentMethod = segmentMethod;
                TimelapseObj{t}.FilterSize = FilterSize;
                TimelapseObj{t}.Timestamp_analysis = timestamp;
                TimelapseObj{t}.alpha_percentile = alpha_percentile;
                TimelapseObj{t}.deg_lbdv_fit = deg_lbdv_fit;
                TimelapseObj{t}.MAX_lbdv_fit_PTS = MAX_lbdv_fit_PTS;
                TimelapseObj{t}.ReferenceCoordinates = RC;
            end
            
        end
        function [XYZ_ell, AER_ell, kap12_ell] = map2Ellipsoid(XYZ,ellipsoidals)
            %% 1. TRANSLATE COORDINATES TO CENTER
            XYZ_cntr = XYZ - ellipsoidals.center;
            
            %% 2. ROTATE COORDINATES TO LOCAL FRAME
            rotMatrix = ellipsoidals.rotMatrix;
            XYZ_local = (rotMatrix\XYZ_cntr')';
            
            %% 3. compute the analagous ellipsoidal coordinates
            % We use a "thread" construction discovered by Staude in 1882.
            % Hilbert and Cohn-Vossen 1999, pp. 19-22
            a = ellipsoidals.semiAxesLengths(1);
            b = ellipsoidals.semiAxesLengths(2);
            c = ellipsoidals.semiAxesLengths(3);
            
            xo = XYZ_local(:,1);
            yo = XYZ_local(:,2);
            zo = XYZ_local(:,3);
            
            u_ref = atan((a*yo)./(b*xo));
            u = u_ref;
            Q1 = (xo>0)&(yo>0);
            Q2 = (xo<0)&(yo>0);
            Q3 = (xo<0)&(yo<0);
            Q4 = (xo>0)&(yo<0);
            
            u(Q1)=u_ref(Q1);
            u(Q2)=u_ref(Q2)+pi;
            u(Q3)=u_ref(Q3)+pi;
            u(Q4)=u_ref(Q4)+2*pi;
            
            cosu = cos(u);
            t = sqrt(1./((zo./c).^2+(xo./(a*cosu)).^2));
            cosv = zo.*t/c;
            sinv = xo.*t./(a*cosu);
            sinu = yo.*t./(b*sinv);
            
            xEllLoc = a*cosu.*sinv;
            yEllLoc = b*sinu.*sinv;
            zEllLoc = c*cosv;
            
            XYZ_ellLoc = [xEllLoc,yEllLoc,zEllLoc];
            XYZ_ell_cntrd = (rotMatrix*XYZ_ellLoc')';
            [azi_ell,elv_ell,rad_ell] = cart2sph(XYZ_ell_cntrd(:,1),XYZ_ell_cntrd(:,2),XYZ_ell_cntrd(:,3));
            AER_ell = [azi_ell,elv_ell,rad_ell];
            XYZ_ell = XYZ_ell_cntrd + ellipsoidals.center;
            %% 4. compute the analytical curvatures of the ellipsoidal coordinates
            cos2v = cosv.^2-sinv.^2;
            cos2u = cosu.^2-sinu.^2;
            
            % OLDER VERSION - CONTAINED ERROR!
            %H = (a*b*c*(3*(a^2+b^2)+2*c^2+(a^2+b^2-2*c^2)*...
            %     cos2v-2*(a^2-b^2)*cos2u.*sinv.^2)./...
            %    (8*(a^2*b^2*cosv.^2+c^2*(b^2*cosu.^2+a^2*sinu.^2).*sinv.^2).^(3/2)));
            %K = real(((a*b*c)^2)./((a*b*cosv).^2+(c^2).*...
            %    ((b*cosu).^2+(a*sinu).^2).*sinv.^2).^2);
            
            % SEPT 10 2020 VERSION
            H_upstairs = a*b*c*( 3*(a^2 + b^2) + 2*c^2 + ...
                (a^2 + b^2 - 2*c^2).*cos2v - 2*(a^2 - b^2).*cos2u.*(sinv.^2));
            H_downstairs = 8*(a^2*b^2*(cosv).^2 + ...
                c^2*(b^2*cosu.^2 + a^2*sinu.^2).*sinv.^2).^(3/2);
            H_new = H_upstairs./H_downstairs;
            
            K_upstairs = (a^2)*(b^2)*(c^2);
            K_downstairs = (a^2*b^2*cosv.^2 + c^2*(b^2*cosu.^2 + a^2*sinu.^2).*sinv.^2).^2;
            K_new = K_upstairs./K_downstairs;
            
            if any(H_new.^2 < K_new)
               error('H_new.^2 < K_new') 
            end
            
            k1 = H_new + sqrt(H_new.^2-K_new);
            k2 = H_new - sqrt(H_new.^2-K_new);
            kap12_ell = [k1,k2];
            
        end
        function [X_mesh,Y_mesh,Z_mesh,R_mesh,Q_mesh] = meshPointCloud(XYZ,n,Q)
            center = nanmean(XYZ);
            indKeep = isfinite(XYZ(:,1)); % Remove non-finite values
            xC = XYZ(indKeep,1) - center(1);
            yC = XYZ(indKeep,2) - center(2);
            zC = XYZ(indKeep,3) - center(3);
            
            [azi,elv,rad] = cart2sph(xC,yC,zC);
            aziInt = [azi-2*pi ; azi ; azi+2*pi];
            elvInt = [elv ; elv ; elv];
            radInt = [rad ; rad ; rad];
            
            fR = scatteredInterpolant(aziInt,elvInt,radInt);
            
            U = pi*ones(n+1,1)*(-n:2:n)/n;
            V = (pi/2)*(-n:2:n)'*ones(1,n+1)/n;
            R = fR(U,V);
            
            [X_mesh,Y_mesh,Z_mesh] = sph2cart(U,V,R);
            R_mesh = R;
            if nargin == 3
                QInt = [Q;Q;Q];
                fQ = scatteredInterpolant(aziInt,elvInt,QInt);
                Q_mesh = fQ(U,V);
            end
        end
        function [Xn_out,Xq_out,D,PatchCenter,OrientMatrix] = orientPatch(Xn_in,Xq_in,Xcenter)
            if isnan(sum(Xq_in)) || ~isfinite(sum(Xq_in))
                error('Query point must be a finite 1x3 array')
            end
            indAccept = ~isnan(Xn_in(:,1)); % Check for NAN values
            x = Xn_in(indAccept,1);
            y = Xn_in(indAccept,2);
            z = Xn_in(indAccept,3);
            xo = mean(x);
            yo = mean(y);
            zo = mean(z);
            PatchCenter = [xo yo zo];
            X = [x-xo,y-yo,z-zo];
            Xq = Xq_in - [xo,yo,zo];
            Xct = Xcenter - [xo,yo,zo];
            n = length(X);
            S = 1/(n-1)*(X'*X);
            [V,D] = eig(S);
            
            OrientMatrix = [V(:,3),V(:,2),V(:,1)]; % reverse vector order so surface normal is in z
            
            Yq = Xq*OrientMatrix;
            YCenter = Xct*OrientMatrix;
            
            if (Yq(3) - YCenter(3) > 0)
                FlipUpsideDown = [1 0 0; 0 -1 0; 0 0 -1]; % Rotates 180 deg about x-axis
                OrientMatrix = OrientMatrix*FlipUpsideDown;
                Yq = Xq*OrientMatrix;
                YCenter = Xct*OrientMatrix;
            end
                
            Y = X*OrientMatrix;
            
            Xq_out = Yq;
            Xn_out = Y;
            
            %x_out = Xn_out(:,1);
            %y_out = Xn_out(:,2);
            %z_out = Xn_out(:,3);
            %close all
            %scatter3(x_out,y_out,z_out,'or')
              
        end
        function [XYZ,Err,FitParams] = parFitTraces(traceStarts,traceEnds,F,fluorescence,edgeFit,testing)
            if nargin == 5
                testing = 0; % Change to 1 to display plots
            end
            N = length(traceEnds);
            XYZ = nan(N,3);
            Err = nan(N,1);
            if edgeFit
                switch fluorescence
                    case 'Fluorescent Interior'
                        numFitPar = 5;
                    case 'Fluorescent Surface'
                        numFitPar = 4;
                    otherwise
                        error('Unexpected value for ''fluorescence''. Expected ''Fluorescent Interior'' or ''Fluorescent Surface''')
                end
            else
                numFitPar = 1;
            end
            FitParams = nan(N,numFitPar);
            samplingLength = 1;
            % This par loop is quicker than the for loop in
            % seg3D.fitTaces(...)
            parfor n = 1:N
                fitPar = []; %#ok<NASGU>
                rStart = traceStarts(n,:);
                rEnd = traceEnds(n,:);
                [traceCoords,sampVec] = ...
                    seg3D.generateLineCoords(rStart, rEnd, samplingLength);
                [t,Int] = seg3D.getTraceToFit(traceCoords,F);
                if edgeFit
                    [startP,t_trimmed,Int_trimmed] = seg3D.estFitParameters(t,Int,fluorescence,testing);
                    [fitPar,~,~,FVU] = seg3D.fitBoundary(t_trimmed,Int_trimmed,startP,fluorescence);
                    XYZ(n,:) = fitPar(2)*sampVec + rStart;
                    Err(n) = FVU;
                    FitParams(n,:) = fitPar;
                else
                    switch fluorescence
                        case 'Fluorescent Interior'
                            dInt = abs(diff(Int));
                            fitPar = t(dInt==max(dInt));
                        case 'Fluorescent Surface'
                            fitPar = t(Int==max(Int));
                        otherwise
                            error('Unexpected value for ''fluorescence''. Expected ''Fluorescent Interior'' or ''Fluorescent Surface''')
                    end
                    XYZ(n,:) = fitPar*sampVec + rStart;
                    Err(n) = 0;
                    FitParams(n,:) = fitPar;
                end
            end
        end
        function intArray3D_out = processImage(intArray3D_in,filterSize,vsx,vsz)
            % Processes image with 3D Gaussian filter
            % Consider resampling 3D tif with isotropic voxel size
            fprintf('     Resampling in Z... ');
            tic
            V = seg3D.resample3D(intArray3D_in,vsx,vsz);
            tRS = toc;
            if filterSize <= 0
                fprintf('     Skipping 3D Gaussian filter (size %d < 0)...',filterSize);
                intArray3D_out = V;
            else
                fprintf('(%3.1f seconds)\n',tRS);
                tic
                fprintf('     Applying 3D Gaussian filter (size %d)...',filterSize);
                gaussPatchSize = 2*(floor((9*filterSize)/2)+0.5);
                intArray3D_out = smooth3(V,'gaussian',...
                    gaussPatchSize,filterSize); % TAKES ALMOST A MINUTE
            end
            tGF = toc;
            fprintf('(%3.1f seconds)\n',tGF);
            % Preview page of filtered tif
            close all
            [~,~,P] = size(intArray3D_out);
            p = floor(P/2);
            imshow(intArray3D_out(:,:,p),[])
            
            
        end
        function [XYZ_out] = ResampleSurface(XYZ_in,samplingLength)
            tic
            fprintf('       Resampling surface...')
            indKeep = ~isnan(XYZ_in(:,1))&isfinite(XYZ_in(:,1)); % Exclude nan and Inf values;
            if sum(indKeep) < length(XYZ_in)
                warning('Inf or Nan values found in coordinates. Removed before resampling.')
            end
            XYZ_unique = unique(XYZ_in(indKeep,:),'rows');
            
            x = XYZ_unique(:,1);
            y = XYZ_unique(:,2);
            z = XYZ_unique(:,3);
            
            center = [nanmean(x), nanmean(y), nanmean(z)];
            
            xCntrd = x - center(1);
            yCntrd = y - center(2);
            zCntrd = z - center(3);
            
            [azi, elv, rad] = cart2sph(xCntrd,yCntrd,zCntrd);
            
            avgRad = mean(rad);
            surfAreaEst = 4*pi*avgRad^2;
            N = ceil(surfAreaEst/(samplingLength^2));
            
%             %% INTERPOLATE RADIAL COORDINATES
%             azi4interp = [azi-2*pi;azi;azi+2*pi];
%             elv4interp = [elv;elv;elv];
%             rad4interp = [rad;rad;rad];
%             F = scatteredInterpolant(azi4interp, elv4interp, rad4interp);
%             [x,y,z] = seg3D.getNormSpiralCoords(N);
%             [azi, elv, ~] = cart2sph(x,y,z);
%             rad = F(azi,elv);
%             [x,y,z] = sph2cart(azi, elv, rad);
            
            %% INTERPOLATE CARTESIAN COORDINATES
            azi4interp = [azi-2*pi;azi;azi+2*pi];
            elv4interp = [elv;elv;elv];
            x4interp = [xCntrd;xCntrd;xCntrd];
            y4interp = [yCntrd;yCntrd;yCntrd];
            z4interp = [zCntrd;zCntrd;zCntrd];
            Fx = scatteredInterpolant(azi4interp, elv4interp, x4interp);
            Fy = scatteredInterpolant(azi4interp, elv4interp, y4interp);
            Fz = scatteredInterpolant(azi4interp, elv4interp, z4interp);
            [x,y,z] = seg3D.getNormSpiralCoords(N);
            [azi, elv, ~] = cart2sph(x,y,z);
            Fx.Method = 'natural';
            Fy.Method = 'natural';
            Fz.Method = 'natural';
            x = Fx(azi,elv);
            y = Fy(azi,elv);
            z = Fz(azi,elv);
            
            xNew = x + center(1);
            yNew = y + center(2);
            zNew = z + center(3);
            
            XYZ_out = [xNew,yNew,zNew];
            tRS = toc;
            fprintf('(%3.1f seconds)\n',tRS)
        end
        function [V_out] = resample3D(V_in,vsx,vsz)
            [M,N,P] = size(V_in);
            V_in = double(V_in); % convert from 16-bit to double
            
            x = 1:M;
            y = 1:N;
            z = (0:(P-1))*(vsz/vsx); % rescales vertical sectioning
            
            F_int = griddedInterpolant({x,y,z},V_in, 'linear', 'none');
            P_new = floor((vsz/vsx)*(P-1));
            
            xNew = x;
            yNew = y;
            zNew = 1:P_new;
            
            V_out = F_int({xNew,yNew,zNew}); % resamples voxels at pixel length
        end
        function [x_out,y_out] = rotateCoordsXYPlane(x_in,y_in,theta)
            x_in = x_in(:)'; % ensures x_in is a row array
            y_in = y_in(:)'; % ensures y_in is a row array
            RotMat = [cos(theta), -sin(theta);
                sin(theta), cos(theta)];
            X = [x_in;y_in];
            Y = RotMat*X;
            
            x_out = Y(1,:);
            y_out = Y(2,:);
        end
        function [TimelapseObj] = Timelapse()
            TimelapseObj = seg3D.makeTimelapseObj(); % Prompts user to provide path to tiff stack
            numOfTimesteps = length(TimelapseObj);
            switch TimelapseObj{1}.segmentMethod
                case 'Local Edge Fit - Parallelized'
                    delete(gcp('nocreate')) % Removes existing parallel pools
                    poolObj = parpool(); % Creates new parallel pool
            end
            for n = 1:numOfTimesteps
                fprintf('Analyzing Timestep %d of %d ...\n',n,numOfTimesteps)
                tic % Begin timer
                TimelapseObj{n} = TimelapseObj{n}.getCoordinates(); % Determines surface coordinates
                TimelapseObj{n} = TimelapseObj{n}.getEllipsoid();
                TimelapseObj{n} = TimelapseObj{n}.SaveResults();
                TimelapseObj{n} = TimelapseObj{n}.getCurvatures(); % Determines surface mean curvatures
                TimelapseObj{n} = TimelapseObj{n}.getFeatures();
                if n == numOfTimesteps || n == 1
                    TimelapseObj{n}.displayPoints();
                end
                TimelapseObj{n}.ComputationTime = toc;
                TimelapseObj{n} = TimelapseObj{n}.SaveResults();
            end
            switch TimelapseObj{1}.segmentMethod
                case 'Local Edge Fit - Parallelized'
                    delete(poolObj) % removes parallel pool
            end
        end
        function [TimelapseObj] = Timelapse_PtCloud()
            error('Not currently a feature')
        end
    end
    methods
        function obj = displayGooseberry(obj)
            close all
            [hRadScatter,hSurfScatter] = obj.displaySurface();
            hRadScatter.FaceAlpha = 0.5;
            hSurfScatter.FaceAlpha = 0.5;
            N = 300;
            [NormVecs,XYZ,H] = obj.resampleNormals(N);
            X = XYZ(:,1);
            Y = XYZ(:,2);
            Z = XYZ(:,3);
            %scatter3(X,Y,Z,[],H,'.')
            U = -(H-mean(H)).*NormVecs(:,1);
            V = -(H-mean(H)).*NormVecs(:,2);
            W = -(H-mean(H)).*NormVecs(:,3);
            hold on
            inward = H < mean(H);
            hInward = quiver3(X(inward),Y(inward),Z(inward),U(inward),V(inward),W(inward));
            hInward.LineWidth = 1;
            hInward.Marker = '.';
            hInward.MarkerEdgeColor = 'k';
            hInward.Color = 'k';
            outward = H >= mean(H);
            hOutward = quiver3(X(outward),Y(outward),Z(outward),U(outward),V(outward),W(outward));
            hOutward.LineWidth = 1;
            hOutward.Marker = '.';
            hOutward.MarkerEdgeColor = 'r';
            hOutward.Color = 'r';
            set(gca,'visible','off');
        end
        function [hRadScatter,hCurvScatter] = displayPoints(obj,opt1)
            X = obj.Coordinates(:,1);
            Y = obj.Coordinates(:,2);
            Z = obj.Coordinates(:,3);
            R = sqrt(sum([X - mean(X), Y - mean(Y), Z - mean(Z)].^2,2));
            H = obj.Curvatures;
            if nargin == 1
                opt1 = '';
            end
            switch opt1
                case 'close'
                    close all
                case 'hold'
                    hold on
                otherwise
                    hold off
            end    
            X_um = X*obj.vsx;
            Y_um = Y*obj.vsx;
            Z_um = Z*obj.vsx;
            R_um = R*obj.vsx;
            H_inv_um = H./obj.vsx;
            
            figure(1)
            hRadScatter = scatter3(X_um,Y_um,Z_um,[],R_um,'.');
            xlabel('x (microns)')
            ylabel('y (microns)')
            zlabel('z (microns)')
            cb = colorbar();
            cb.Label.String = 'Radius (microns)';
            axis equal
            figure(2)
            hCurvScatter = scatter3(X_um,Y_um,Z_um,[],H_inv_um,'.');
            xlabel('x (microns)')
            ylabel('y (microns)')
            zlabel('z (microns)')
            cb = colorbar();
            cb.Label.String = 'Mean Curvature (1/microns)';
            axis equal
        end
        function [hRadSurf,hCurvSurf] = displaySurface(obj)
            n = ceil(max(range(obj.Coordinates))*2*pi);
            [X,Y,Z,R,H] = obj.meshPointCloud(obj.Coordinates,n,obj.Curvatures);
            if nargin == 1
                opt1 = '';
            end
            switch opt1
                case 'close'
                    close all
                case 'hold'
                    hold on
                otherwise
                    hold off
            end
            figure(1)
            hRadSurf = surf(X,Y,Z,R);
            hRadSurf.EdgeColor = 'none';
            axis equal
            figure(2)
            hCurvSurf = surf(X,Y,Z,H);
            hCurvSurf.EdgeColor = 'none';
            axis equal
        end
        function obj = estimateCurvatures(obj)
            %circumference = obj.findCircumference(x,y);
            obj.SurfaceArea = obj.findSurfaceArea(obj.Coordinates);
            [XYZ_ell, AER_ell, kap12_ell] = obj.map2Ellipsoid(obj.Coordinates,obj.Ellipsoidals);
            %[theta, rho] = cart2pol(xCntrd,yCntrd);
            %[x_ell,y_ell,thet_ell,rho_ell,kappa_ell] = ...
            %    obj.map2Ellipse(x,y,ellipticals);
            
            if obj.testing
                x = obj.Coordinates(:,1);
                y = obj.Coordinates(:,2);
                z = obj.Coordinates(:,3);
                
                x_ell = XYZ_ell(:,1);
                y_ell = XYZ_ell(:,2);
                z_ell = XYZ_ell(:,3);
                
                azi_ell = AER_ell(:,1);
                elv_ell = AER_ell(:,2);
                rad_ell = AER_ell(:,3);
                
                H_ell = mean(kap12_ell,2);
                
                figure(1)
                hold off
                scatter3(x_ell,y_ell,z_ell,H_ell,'.r')
                hold on
                scatter3(x,y,z,'.k')
                axis equal
                
                figure(2)
                hold off
                scatter3(azi_ell,elv_ell,H_ell,'.')
            end
            
            obj.k1 = kap12_ell(:,1); % replace with calculated values
            obj.k2 = kap12_ell(:,2); % replace with calculated values
        end
        function obj = filterPatches(obj)
            N = obj.NumCoords;
            H_in = obj.Curvatures;
            H_out = nan(N,1);
            obj = obj.findNeighborIndices;
            indices = obj.Indices();
            for n=1:N
                indNbrs = indices{n};
                H_patch = H_in(indNbrs);
                H_out(n) = nanmedian(H_patch); % updated 2020-09-14 to nanmedian to prevent surface erosion.
            end
            obj.Curvatures = H_out;
        end
        function obj = findNeighborIndices(obj)
            obj = obj.removeNanCoords();
            N = obj.NumCoords;
            [obj.PatchRadii,obj.NS_kd] = obj.getPatchRadii();
            queryPoints = obj.Coordinates();
            obj.Indices = cell(N,1); % This takes a lot of memory. Consider clearing values later after computation is complete
            obj.Distances = cell(N,1); % This takes a lot of memory. Consider clearing values later after computation is complete
            for n = 1:N
                [idx,dist]=rangesearch(obj.NS_kd,queryPoints(n,:),obj.PatchRadii(n));
                obj.Indices{n} = idx{1};
                obj.Distances{n} = dist{1};
            end
        end
        function obj = findNeighborIndicesMedFltr(obj)
            N = obj.NumCoords;
            patchRadii = obj.PatchRadiusForMedianFilter*ones(N,1);
            queryPoints = obj.Coordinates();
            obj.Indices = cell(N,1); % This takes a lot of memory. Consider clearing values later after computation is complete
            obj.Distances = cell(N,1); % This takes a lot of memory. Consider clearing values later after computation is complete
            for n = 1:N
                [idx,dist]=rangesearch(obj.NS_kd,queryPoints(n,:),patchRadii(n));
                obj.Indices{n} = idx{1};
                obj.Distances{n} = dist{1};
            end
        end
        function obj = fitFirstPatches(obj)
            fprintf('Estimating curvatures and building kd-tree ...\n')
            obj = obj.estimateCurvatures();
            obj.NS_kd = KDTreeSearcher(obj.Coordinates);
            obj = obj.findNeighborIndices();
            switch obj.segmentMethod
                case {'Local Edge Fit - Parallelized', 'Quick Edge Detection - Parallelized'}
                    obj = obj.parFitPatches();
                case {'Local Edge Fit', 'Quick Edge Detection'}
                    obj = obj.fitPatches();
                otherwise
                    error('Unexpected value for ''segmentMethod''')
            end
        end
        function obj = fitIterativePatches(obj)
            N = obj.NumberOfIterations;
            meanSqrdRelChangeCurv = nan(N,1);
            for n = 1:N
                if n >= N - 1
                    obj.proj2Fit = 1;
                end
                fprintf('Fitting Patches (Iteration No. %d of %d) ...\n',n,N)
                obj = obj.findNeighborIndices();
                H_old = obj.Curvatures;
                switch obj.segmentMethod
                    case {'Local Edge Fit - Parallelized','Quick Edge Detection - Parallelized'}
                        obj = obj.parFitPatches();
                    case {'Local Edge Fit','Quick Edge Detection'}
                        obj = obj.fitPatches();
                    otherwise
                        error('Unexpected value for ''segmentMethod''.')
                end
                H_new = obj.Curvatures;
                H_relChange = (H_new-H_old)./H_new;
                meanSqrdRelChangeCurv(n) = nanmean(H_relChange.^2);
                obj.CurrentIteration = n;
                % Remove nan coordinates, etc.
                ind = ~isnan(H_new);
                obj.Coordinates = obj.Coordinates(ind,:);
                obj.Curvatures = obj.Curvatures(ind,:);
                obj.k1 = obj.k1(ind,:);
                obj.k2 = obj.k2(ind,:);
            end
        end
        function obj = fitPatches(obj)
            N = obj.NumCoords;
            coords = obj.Coordinates;
            center = mean(coords);
            indices = obj.Indices();
            H = nan(N,1);
            kap1 = nan(N,1);
            kap2 = nan(N,1);
            MinimumPerPatch = 6;
            norms = nan(N,3);
            for n=1:N
                queryPoint = coords(n,:);
                indNbrs = indices{n};
                neighbors = coords(indNbrs,:);
                emptyPatch = length(neighbors) < MinimumPerPatch;
                if emptyPatch
                    coords(n,:) = [nan nan nan];
                    H(n) = nan;
                    kap1(n) = nan;
                    kap2(n) = nan;
                else
                    [X,Xq,~,PatchCenter,OrientPatch] = obj.orientPatch(neighbors,queryPoint,center);
                    x = X(:,1);
                    y = X(:,2);
                    z = X(:,3);
                    [theta,~,h] = obj.fitPoly2(x,y,z);
                    [H(n),kap1(n),kap2(n)] = obj.curvFromPoly2Fit(theta,Xq);
                    if obj.proj2Fit
                        X_proj_local = [Xq(1:2),h(1)];
                        XFit = X_proj_local/OrientPatch + PatchCenter;
                        coords(n,:) = XFit;
                        norms(n,:) = OrientPatch(:,3)';
                    end
                end
            end
            obj.Normals = norms;
            obj.Coordinates = coords;
            obj.Curvatures = H;
            obj.k1 = kap1;
            obj.k2 = kap2;
        end
        function obj = getCoordinates(obj)
            fprintf('   Locating surface coordinates from tif stack...\n')
            intArray3D = obj.loadImages(); % load images into array
            V = obj.processImage(intArray3D,obj.FilterSize,obj.vsx,obj.vsz); %
            sampLen = sqrt(1/obj.SurfDens); % In Pixels
            [M,N,P] = size(V);
            boxLimits = [1 1 1; M N P];
            %[rows,cols,pages] = size(V);
            %V = cat(1,zeros(1,cols,pages),V,zeros(1,cols,pages));
            %V = cat(2,zeros(rows+2,1,pages),V,zeros(rows+2,1,pages));
            %V = cat(3,zeros(rows+2,cols+2,1),V,zeros(rows+2,cols+2,1));
            %[M,N,P] = size(V);
            %boxLimits = [2 2 2; M-1 N-1 P-1];
            
            [X,Y,Z] = ndgrid(1:M,1:N,1:P);
            F_int = griddedInterpolant(X,Y,Z,V, 'cubic', 'nearest');
            patchRadius = obj.PatchRadiusForIQRFilter;
            switch obj.modelShape
                case 'Ellipsoid'
                    [estCenter,semiAxesLengths,rotMatrix] = obj.estEllipsoidFromVolume(V,X,Y,Z,obj.fluorescence,obj.testing);
                    modelParams.semiAxesLengths = semiAxesLengths;
                    modelParams.center = estCenter;
                    modelParams.rotMatrix = rotMatrix;
                case 'Sphere'
                    [estCenter,estRadius] = obj.estCenAndRad3D(V,X,Y,Z,obj.fluorescence,obj.testing);
                    modelParams.center = estCenter;
                    modelParams.radius = estRadius;
                otherwise
                    error('Unexpected value for ''modelShape''.')
            end
            switch obj.segmentMethod
                case 'Local Edge Fit'
                    parallelize = 0;
                    edgeFit = 1;
                case 'Local Edge Fit - Parallelized'
                    parallelize = 1;
                    edgeFit = 1;
                case 'Quick Edge Detection'
                    parallelize = 0;
                    edgeFit = 0;
                case 'Quick Edge Detection - Parallelized'
                    parallelize = 1;
                    edgeFit = 0;
                otherwise
                    error('Unexpected value for ''segmentMethod''.')
            end
            [XYZ,Err] = obj.getXYZfromVolume(F_int,boxLimits,obj.modelShape,modelParams,sampLen,obj.fluorescence,patchRadius,parallelize,edgeFit);
            obj.Coordinates = XYZ;
            obj.CoordErrors = Err;
            if obj.testing
                close all
                [~,~,P] = size(V);
                for p = 1:P
                    inPlane = (XYZ(:,3) >= (p-1))&(XYZ(:,3) < p);
                    XY = XYZ(inPlane,1:2);
                    imshow(V(:,:,p),[]);
                    hold on
                    plot(XY(:,2),XY(:,1),'.r')
                    pause(0.1)
                end
            end
        end
        function obj = getCurvatures(obj)
            % Determines local mean curvatures from coordinates
            obj = obj.fitFirstPatches();
            obj = obj.fitIterativePatches();
            obj = obj.medianFilterCurvatures();
        end
        function obj = getEllipsoid(obj)
            % Extracts information about ellipsoidal deformation of object
            switch obj.segmentMethod
                case {'Local Edge Fit - Parallelized', 'Quick Edge Detection - Parallelized'}
                    parallelize = 1;
                otherwise
                    parallelize = 0;
            end
            
            [estCenter,semiAxesLengths,rotMatrix] = obj.estEllipsoidFromCoordinates(obj.Coordinates,parallelize);
            
            x = obj.Coordinates(:,1);
            y = obj.Coordinates(:,2);
            z = obj.Coordinates(:,3);
            
            obj.Ellipsoidals.center = estCenter;
            obj.Ellipsoidals.semiAxesLengths = semiAxesLengths;
            obj.Ellipsoidals.rotMatrix = rotMatrix;
            
            
            if obj.testing
                numCoords = length(x);
                [XYZ_fit,~] = seg3D.ellipsoidCoords(semiAxesLengths,numCoords,estCenter,'Rotation Matrix',rotMatrix);
                xFit = XYZ_fit(:,1);
                yFit = XYZ_fit(:,2);
                zFit = XYZ_fit(:,3);
                close all
                scatter3(x,y,z,'.k')
                hold on
                scatter3(xFit,yFit,zFit,'.r');
                axis equal
            end
        end
        function obj = getFeatures(obj)
            % getFeatures returns the seg3D obj with the Feature property
            % defined
            % Created 13 June 2019
            if isempty(obj.Path2Feat)
                fprintf('Feature analysis was not performed.\n')
            else
                intArray2D = obj.loadImages('feature');
                [M,N] = size(intArray2D);
                [Y,X] = meshgrid(1:N,1:M);
                V = double(intArray2D);
                F_int = griddedInterpolant(X,Y,V, 'cubic', 'none');
                featSigma = obj.FeatureKernelSize;
                featKernel = fspecial('gaussian',4*featSigma,featSigma);
                obj.Features = obj.getFeatFromFrame(F_int,obj.Coordinates,featKernel);
                fprintf('Feature analysis performed.\n')
            end
        end
        function minRp = get.minRp(obj)
            minRp = sqrt(obj.MinNumPerPatch/(pi*obj.SurfDens)); % min patch radius req for ~MinNumPerPatch
        end
        function NumCoords = get.NumCoords(obj)
            [NumCoords,~] = size(obj.Coordinates);
        end
        function Path2Mat = get.Path2Mat(obj)
            ts = sprintf('%.4d',obj.Timestep);
            filename = ['ts',ts,'-Results.mat'];
            Path2Mat = [obj.Path2MatDir,filename];
        end
        function Path2MatDir = get.Path2MatDir(obj)
            methodStamp = obj.segmentMethod;
            dirOutput = dir(obj.Path2Tif);
            tifName = dirOutput.name;
            timeStampStr = obj.Timestamp_analysis;
            OutputFolderName = [tifName(1:end-4),'_analyzed',timeStampStr];
            if ispc
                Path2MatDir = [obj.Path2TifDir,OutputFolderName,'\seg3DResults-',methodStamp,'\'];
            else
                Path2MatDir = [obj.Path2TifDir,OutputFolderName,'/seg3DResults-',methodStamp,'/'];
            end
            if ~exist(Path2MatDir,'dir')
                mkdir(Path2MatDir)
            end
        end
        function [PatchRadii,NS_kd] = getPatchRadii(obj)
            % NOTE: Still need to implement maxRp check
            % Patch radii are obtained from maximal principal curavtures
            kMax = obj.k1(:)'; % k1 = kMax by definition
            lenScale1 = obj.SmallPatchLengthScale; % 1st length scale (small)
            lenScale2 = 2./kMax; % 2/curvature = 2nd length scale
            % Patch radii is the geometric mean of these two length scales
            PatchRadii = sqrt(lenScale1*lenScale2);
            indTooSmall = (PatchRadii<obj.minRp);
            PatchRadii(indTooSmall) = obj.minRp;
            queryPoints = obj.Coordinates;
            N = length(queryPoints);
            NS_kd = KDTreeSearcher(queryPoints);
            %% Make sure patch isn't too big.
            if ~isempty(obj.Curvatures)
                for n=1:N
                    [indNbrs,dists] = rangesearch(NS_kd,queryPoints(n,:),PatchRadii(n));
                    %warning('Consider range of magnitude of k1 and k2.')
                    %H_m = obj.Curvatures(n);
                    k_max = obj.k1(n);
                    k_patch = obj.k1(indNbrs{1});
                    hotSpots = abs(k_patch./k_max) > 2;
                    if sum(hotSpots)
                        rp = min(dists{1}(hotSpots))-0.1;
                        if rp < obj.minRp
                            rp = obj.minRp;
                        end
                        [indNbrs,~] = rangesearch(NS_kd,queryPoints(n,:),rp);
                        if length(indNbrs{1}) < obj.MinNumPerPatch
                            [~,dists] = knnsearch(NS_kd,queryPoints(n,:),'k',obj.MinNumPerPatch);
                            rp = max(dists);
                        end
                        PatchRadii(n) = rp;
                    end
                    %warning('Add final check to ensure minimum number per patch.')
                end
            end
            
        end
        function Path2TifDir = get.Path2TifDir(obj)
            path2tif = obj.Path2Tif;
            if ispc
                slash = '\';
            else
                slash = '/';
            end
            indLastSlash = find(path2tif == slash, 1, 'last' );
            Path2TifDir = path2tif(1:indLastSlash);
        end
        function intArray3D = loadImages(obj)
            fprintf('     Loading Image...')
            tic
            FileTif=obj.Path2Tif;
            InfoImage=imfinfo(FileTif);
            mImage=InfoImage(1).Width;
            nImage=InfoImage(1).Height;
            NumberImages=length(InfoImage);
            TifLink = Tiff(FileTif, 'r');
            
            if isempty(obj.FirstFrame)
                obj.FirstFrame = 1;
            end
            if isempty(obj.LastFrame)
                obj.LastFrame = NumberImages;
            end
            NumberOfFrames = obj.LastFrame - obj.FirstFrame + 1;
            intArray3D=zeros(nImage,mImage,NumberOfFrames,'uint16');
            FrameNumbers = obj.FirstFrame:obj.LastFrame;
            for i = 1:NumberOfFrames
                frameInd = FrameNumbers(i);
                TifLink.setDirectory(frameInd);
                intArray3D(:,:,i)=TifLink.read();
            end
            TifLink.close();
            tLoad = toc;
            fprintf('(%3.1f seconds)\n',tLoad)
        end
        function obj = medianFilterCurvatures(obj)
            medFltrPatchRad = obj.PatchRadiusForMedianFilter;
            if medFltrPatchRad > 0
                fprintf('Median Filtering Curvatures (patch radius %d) ...\n',medFltrPatchRad)
                H_old = obj.Curvatures;
                obj = obj.findNeighborIndicesMedFltr();
                obj = obj.filterPatches();
                H_new = obj.Curvatures;
                H_relChange = (H_new-H_old)./H_new;
            else
                fprintf('No Median Filtering of Curvatures (patch radius %d) ...\n',medFltrPatchRad)
            end
        end
        function obj = parFitPatches(obj)
            N = obj.NumCoords;
            coords = obj.Coordinates;
            center = mean(coords);
            indices = obj.Indices();
            distances = obj.Distances();
            H = nan(N,1);
            rel_H_range_inPatch = nan(N,1);
            kap1 = nan(N,1);
            kap2 = nan(N,1);
            norms = nan(N,3);
            neighbors = cell(N,1);
            MinimumPerPatch = obj.MinNumPerPatch;
            for n=1:N
                indNbrs = indices{n};
                numInPatch = numel(indNbrs);
                if isempty(obj.Curvatures)
                    [XYZ_ell, AER_ell, kap12_ell] = seg3D.map2Ellipsoid(obj.Coordinates,obj.Ellipsoidals);
                    obj.Curvatures = mean(kap12_ell,2);
                    if any(isnan(obj.Curvatures))
                        error('Curvatures should all be defined from ellipsoidal fit. Coding error.')
                    end
                end
                H_old = obj.Curvatures;
                if ~isempty(H_old)
                    inPatchHRange = range(H_old(indNbrs));
                    rel_H_range_inPatch(n) = inPatchHRange./H_old(n);
                    while ~isnan(rel_H_range_inPatch(n)) && rel_H_range_inPatch(n) > 2 && numInPatch > MinimumPerPatch
                        % Locate extreme H measurements
                        warning('Large range of mean curvature within patch.')
                        H_inPatch = H_old(indNbrs);
                        maxH_ind = find(H_inPatch==max(H_inPatch),1,'last');
                        minH_ind = find(H_inPatch==min(H_inPatch),1,'last');
                        % Determine new number in patch required to remove
                        % extreme curvature value. Removing extreme value
                        % shrinks range in patch.
                        numInPatch = max([maxH_ind,minH_ind])-1;
                        if numInPatch < MinimumPerPatch
                            numInPatch = MinimumPerPatch;
                        end
                        indNbrs = indices{n}(1:numInPatch);
                        inPatchHRange = range(H_old(indNbrs));
                        rel_H_range_inPatch(n) = inPatchHRange./H_old(n);
                        
                    end
                    if numInPatch < MinimumPerPatch
                        warning('numInPatch is fewer than MinimumPerPatch.')
                    end
                    indices{n} = indices{n}(1:numInPatch);
                    distances{n} = distances{n}(1:numInPatch);
                    if numel(indices{n}) < MinimumPerPatch
                        warning('numInPatch is fewer than MinimumPerPatch.')
                    end
                end
                neighbors{n} = coords(indNbrs,:);
            end
            obj.Indices = indices;
            obj.Distances = distances;
            project2FittedSurface = obj.proj2Fit;
            
            hardMinimumPerPatch = 6;
            parfor n = 1:N
                queryPoint = coords(n,:);
                emptyPatch = length(neighbors{n}) < hardMinimumPerPatch; 
                if emptyPatch
                    coords(n,:) = [nan nan nan];
                    H(n) = nan;
                    kap1(n) = nan;
                    kap2(n) = nan;
                else
                    [X,Xq,~,PatchCenter,OrientPatch] = seg3D.orientPatch(neighbors{n},queryPoint,center);
                    x = X(:,1);
                    y = X(:,2);
                    z = X(:,3);
                    [theta,~,h] = seg3D.fitPoly2(x,y,z);
                    [H(n),kap1(n),kap2(n)] = seg3D.curvFromPoly2Fit(theta,Xq);
                    if project2FittedSurface
                        X_proj_local = [Xq(1:2),h(1)];
                        XFit = X_proj_local/OrientPatch + PatchCenter;
                        coords(n,:) = XFit;
                        norms(n,:) = OrientPatch(:,3)';
                    end
                end
            end
            obj.Normals = norms;
            obj.Coordinates = coords;
            obj.Curvatures = H;
            obj.k1 = kap1;
            obj.k2 = kap2;
        end
        function obj = removeNanCoords(obj)
            indKeep = ~isnan(obj.Coordinates(:,1));
            XYZ = obj.Coordinates(indKeep,:);
            obj.Coordinates = XYZ;
            XYZ_err = obj.CoordErrors(indKeep,:);
            obj.CoordErrors = XYZ_err;
            if ~isempty(obj.Curvatures)
                H = obj.Curvatures(indKeep);
                obj.Curvatures = H;
            end
            if ~isempty(obj.k1)
                kap1 = obj.k1(indKeep);
                obj.k1 = kap1;
            end
            if ~isempty(obj.k2)
                kap2 = obj.k2(indKeep);
                obj.k2 = kap2;
            end
            if ~isempty(obj.Normals)
                UVW = obj.Normals(indKeep,:);
                obj.Normals = UVW;
            end
        end
        function [NormVecs_out,XYZ_out,H_out] = resampleNormals(obj,n)
            if nargin == 1
                n = input('How many normal vectors to plot? ');
            end
            % Interpolate Coordinates
            [x_sc,y_sc,z_sc] = seg3D.getNormSpiralCoords(n);
            [azi_rs,elv_rs,~] = cart2sph(x_sc,y_sc,z_sc);
            cntr = mean(obj.Coordinates);
            Xcntrd = obj.Coordinates(:,1) - cntr(1);
            Ycntrd = obj.Coordinates(:,2) - cntr(2);
            Zcntrd = obj.Coordinates(:,3) - cntr(3);
            [azi,elv,~] = cart2sph(Xcntrd,Ycntrd,Zcntrd);
            Azi4Int = [azi-2*pi;azi;azi+2*pi];
            Elv4Int = [elv;elv;elv];
            X4Int = [Xcntrd;Xcntrd;Xcntrd];
            Y4Int = [Ycntrd;Ycntrd;Ycntrd];
            Z4Int = [Zcntrd;Zcntrd;Zcntrd];
            Fx = scatteredInterpolant(Azi4Int,Elv4Int,X4Int);
            Fy = scatteredInterpolant(Azi4Int,Elv4Int,Y4Int);
            Fz = scatteredInterpolant(Azi4Int,Elv4Int,Z4Int);
            X_rs = Fx(azi_rs,elv_rs);
            Y_rs = Fy(azi_rs,elv_rs);
            Z_rs = Fz(azi_rs,elv_rs);
            
            % Interpolate Normals
            Nx = obj.Normals(:,1);
            Ny = obj.Normals(:,2);
            Nz = obj.Normals(:,3);
            Nx4Int = [Nx;Nx;Nx];
            Ny4Int = [Ny;Ny;Ny];
            Nz4Int = [Nz;Nz;Nz];
            Fnx = scatteredInterpolant(Azi4Int,Elv4Int,Nx4Int);
            Fny = scatteredInterpolant(Azi4Int,Elv4Int,Ny4Int);
            Fnz = scatteredInterpolant(Azi4Int,Elv4Int,Nz4Int);
            Nx_rs = Fnx(azi_rs,elv_rs);
            Ny_rs = Fny(azi_rs,elv_rs);
            Nz_rs = Fnz(azi_rs,elv_rs);
            
            % Interpolate Curvatures
            H = obj.Curvatures;
            H4Int = [H;H;H];
            Fh = scatteredInterpolant(Azi4Int,Elv4Int,H4Int);
            H_rs = Fh(azi_rs,elv_rs);
            
            % OUTPUTS:
            NormVecs_out = [Nx_rs,Ny_rs,Nz_rs];
            XYZ_out = [X_rs, Y_rs, Z_rs];
            H_out = H_rs;
        end
        function obj = SaveResults(obj)
            obj.Indices = 'property removed after computation to reduce file size'; % Reduces file size dramatically
            obj.Distances = 'property removed after computation to reduce file size'; % Reduces file size dramatically
            obj.NS_kd = 'property removed after computation to reduce file size'; % Reduces file size dramatically
            fullpath = obj.Path2Mat;
            if ~exist(fullpath,'file')
                try
                    save(fullpath,'obj');
                catch
                    save(fullpath,'obj');
                end
            else
                save(fullpath,'obj');
            end
            fprintf('\nResults of analysis have been saved in the following location: \n%s\n',fullpath)
            %% Create justCoordinatesAndCurvaturesNNNN.mat file
            path2MatFile = fullpath;
            pointCloudArray = obj.Coordinates;
            meanCurvatureArray = obj.Curvatures;
            warning('Need to define metadata file ''data''')
            data.H_inv_px = meanCurvatureArray;
            data.H_inv_um = meanCurvatureArray./obj.vsx;
            data.px_sz_um = obj.vsx;
            data.XYZ_px = pointCloudArray;
            data.XYZ_um = pointCloudArray*obj.vsx;
            data.minColCrop_orig_px = nan;
            data.maxColCrop_orig_px = nan;
            data.minRowCrop_orig_px = nan;
            data.maxRowCrop_orig_px = nan;
            RC = obj.ReferenceCoordinates;
            data.drop_center_XYZ_orig_px = [RC.drop_row, RC.drop_col, RC.drop_page];
            data.EK_center_XYZ_orig_px = [RC.EK_row, RC.EK_col, RC.EK_page];
            data.drop_center_XYZ_um = RC.drop_center_XYZ_um;
            data.EK_center_XYZ_um = RC.EK_center_XYZ_um;
            data.binFactorX = nan;
            data.binFactorY = nan;
            data.binFactorZ = nan;
            data.original_px_sz_um = nan;
            data.sequence_ID_string = RC.sequence_ID_string;
            data.timestamp_hrs = RC.timestamp_hrs;
            
            timestamp = obj.Timestamp_analysis;
            [path2CoordsAndCurves] = obj.exportCoordsAndCurves(path2MatFile,pointCloudArray,meanCurvatureArray,data,timestamp);
            %% Create Input_Dict.mat file
            INPUT_DICT.IFT = obj.IFT_milliNewtons_per_meter;
            INPUT_DICT.Pixel_size_microns = obj.vsx;
            INPUT_DICT.alpha_percentile_excl_AnisStress = obj.alpha_percentile;
            INPUT_DICT.alpha_percentile_excl_AnisCellStress = obj.alpha_percentile;
            INPUT_DICT.alpha_percentile_excl_AnisStressLocalMax_m_AnisStressLocalMin = obj.alpha_percentile;
            INPUT_DICT.deg_lbdv_fit = obj.deg_lbdv_fit;
            INPUT_DICT.deg_lbdv_Vol_Int = obj.deg_lbdv_fit;
            INPUT_DICT.deg_fit_Ellipsoid_Deviation_Analysis = obj.deg_lbdv_fit;
            INPUT_DICT.MAX_lbdv_fit_PTS = obj.MAX_lbdv_fit_PTS;
            INPUT_DICT.MAX_lbdv_vol_PTS = obj.MAX_lbdv_fit_PTS;
            INPUT_DICT.EK_center_XYZ_um = obj.ReferenceCoordinates.EK_center_XYZ_um;
            INPUT_DICT.drop_center_XYZ_um = obj.ReferenceCoordinates.EK_center_XYZ_um;
            obj.Path2InputDictMat = obj.generateInputDict(path2CoordsAndCurves,INPUT_DICT);
        end
    end
end