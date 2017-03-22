% Get Mat containing SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,REGIONSIZE,ADJMAT,FEATURE_VECTOR,LABEL_VECTOR,IMG,LABEL_MASK  
% USAGE: ComputeImageCues(InputDir,OutputDir,RegionSize)
%
% INPUT:
%    InputDir  : ImageDirectory
%    OutputDir : OutputDirectory to storemat
%    RegionSize : paramter to regulater the no of super pixel in each image
%    Height: height of image
%    Width: width of image
% OUTPUT:
%       Write mat file in folder
% Author:Sanchit Aggarwal
% Date:17-December-2013 10:29 A.M.

function ComputeImageCues(InputDir,OutputDir,RegionSize,Height,Width,Thresh)
    global IMG LABEL GRAY_IMG SLIC_TIME SLIC_SEGMENT NOSP SLIC_BOUNDARIES REGIONPROP ...
        SLIC_MASK ADJ_TIME ADJMAT FEATURE_VECTOR LABEL_VECTOR FEATURE_TIME X Y ...
        POS_DENSITY_IMAGE POS_DENSITY_TIME HEIGHT WIDTH FLOOR_BOUNDARY_MAP THRESHOLD FLOOR_BOUNDARY_TIME ...
        VL HL IL FPolyX FPolyY PolyX PolyY lines bandwidth;
    addpaths;
    time = tic;
    fnName = 'ComputeImageFeatures:';
    msg = '----------begins----------';
    fprintf('\n%s %s',fnName,msg);
    if ~exist(InputDir,'dir')
       error('\n%s',msg);
    else
        ImagesDir = [InputDir '/Images/'];
        LabelsDir = [InputDir '/Labels/'];
    end
    % Create Output Dir
    if ~exist(OutputDir,'dir')
        mkdir(OutputDir);
    end
    Filename = [OutputDir '/TimeAnalysis'];  
    fp=fopen(Filename,'a+');
    fprintf(fp,'ID,Name,Height,Width,Thresh,RegionSize,SLIC_TIME,ADJ_TIME,FEATURE_TIME,POS_DENSITY_TIME,FLOOR_BOUNDARY_TIME,TOTAL_TIME,PROCESSING_TIME,Bandwidth-1,Bandwidth-2\n');
    
    fclose(fp);
    Images = dir([ImagesDir '*' '.jpg']);
    N = size(Images,1);    % Total Training Images   
    fprintf('\nTotal Images loaded : %d\n',N);
    if nargin<4
        HEIGHT=512;
        WIDTH=512;
        THRESHOLD = 0.6;
    else
        HEIGHT=Height;
        WIDTH=Width;
        THRESHOLD = Thresh;
    end
    Regularizer = 0.01;     % SuperPixel Compactness parameter
    for ID = 1:N
        PROCESSING_TIME=tic;
        fprintf('\n---%d %s----',ID,Images(ID).name);
        IMG = imread([ImagesDir Images(ID).name]);
        IMG = imresize(IMG,[HEIGHT WIDTH]);
        LABEL = load([LabelsDir Images(ID).name(1:end-3) 'mat']);
        LABEL = LABEL.mask;
        LABEL = imresize(LABEL,[HEIGHT WIDTH],'nearest');
        
        %Computing SLIC
        ComputeSLIC(RegionSize,Regularizer);
        %Comput AdjMAt
        ComputeAdjMat;
        %Compute Superpixel Feature_Vector and Label_Vector
        ComputeSPFeatures;
        %Compute Position Density Map
        ComputePositionDensity;
        %Compute FloorBoundary Map
        ComputeFloorBoundary;
        
        
        TOTAL_TIME=SLIC_TIME+ADJ_TIME+FEATURE_TIME+POS_DENSITY_TIME+FLOOR_BOUNDARY_TIME;
        PROCESSING_TIME=toc(PROCESSING_TIME);
        % save time analysis of each image
        fp=fopen(Filename,'a+');
        fprintf(fp,'%d,%s,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'...
            ,ID,Images(ID).name,HEIGHT,WIDTH,THRESHOLD,RegionSize,SLIC_TIME,ADJ_TIME,FEATURE_TIME,POS_DENSITY_TIME,FLOOR_BOUNDARY_TIME,TOTAL_TIME,PROCESSING_TIME,bandwidth(1),bandwidth(2));
        fclose(fp);
        save([OutputDir Images(ID).name(1:end-3) 'mat'],'IMG','LABEL',...
            'GRAY_IMG','SLIC_TIME','SLIC_SEGMENT','NOSP','SLIC_BOUNDARIES', ...
        'SLIC_MASK','ADJ_TIME','ADJMAT','FEATURE_VECTOR','LABEL_VECTOR','FEATURE_TIME', ...
        'POS_DENSITY_IMAGE','POS_DENSITY_TIME','HEIGHT','WIDTH','X','Y',...
        'FLOOR_BOUNDARY_MAP','THRESHOLD','FLOOR_BOUNDARY_TIME','REGIONPROP', ...
        'VL','HL','IL','FPolyX','FPolyY','PolyX','PolyY','lines','bandwidth');
    end
    toc(time);
end

%% Compute SLIC
function ComputeSLIC(RegionSize,Regularizer)
    global IMG SLIC_TIME SLIC_SEGMENT NOSP SLIC_BOUNDARIES;
    IMG = im2double(IMG);
    SLIC_TIME = tic;
    SLIC_SEGMENT = vl_slic(single(IMG), RegionSize, Regularizer);
    SLIC_TIME = toc(SLIC_TIME);
    % check whether # of segments equal to max segment if not correct the
    uniquelabel = unique(SLIC_SEGMENT);
    NOSP = numel(uniquelabel); % # of unique label
    maxlabel = max(SLIC_SEGMENT(:)) + 1;  %slic index starts with 0
    if NOSP == maxlabel
        fprintf('\ncorrect segments');
        % superpixel indexing starts with 0, convet pixels labelled 0 to max.
        SLIC_SEGMENT(SLIC_SEGMENT == 0) =  maxlabel;
    else
        fprintf('\ncorrecting segments maxlabel:%d nunqlabel:%d',maxlabel,NOSP);
        segments = zeros(size(SLIC_SEGMENT),'uint32');
        for j = 1:NOSP
            spval = uniquelabel(j);
            pix = SLIC_SEGMENT == spval;
            segments(pix) = j;
        end
        SLIC_SEGMENT = segments;
    end
    SLIC_BOUNDARIES = rangefilt(SLIC_SEGMENT) > 0;
end

%% Compute SLIC Adjacency Matrix
function ComputeAdjMat
    global SLIC_SEGMENT NOSP ADJ_TIME ADJMAT;
    ADJ_TIME = tic;
    % calculate the 8 neighbour matrix for each pixel
    north = SLIC_SEGMENT([1, 1 : (end -1)], :);
    south = SLIC_SEGMENT([2 : end, end], :);
    east = SLIC_SEGMENT(:, [2 : end, end]);
    west = SLIC_SEGMENT(:, [1, 1 : (end - 1)]);
    
    northWest = SLIC_SEGMENT;
    northWest(2 : end, 2 : end) = SLIC_SEGMENT(1 : end - 1, 1 : end -1);
    
    northEast = SLIC_SEGMENT;
    northEast(2 : end, 1 : (end - 1)) = SLIC_SEGMENT(1 : (end - 1), 2 : end);
    
    southEast = SLIC_SEGMENT;
    southEast(1 : (end - 1), 1 : (end - 1)) = SLIC_SEGMENT(2 : end, 2 : end);
    
    sowthWest = SLIC_SEGMENT;
    sowthWest(1 : (end - 1), 2 : end) = SLIC_SEGMENT(2 : end, 1 : (end - 1));

    neighbours = [northWest(:), north(:), northEast(:), east(:), southEast(:), south(:), ...
                            sowthWest(:), west(:)];
    
    % construct adjacency matrix
    ADJMAT = false(NOSP,NOSP);
    % produce a labels x 1 cell with neighbours for each label
    %%regionNeighbour = arrayfun(@(x) unique(neighbours(segments == x,:)), labels,'UniformOutput', false);
    fprintf('\nCreating Adjacency Matrix');
    for  i = 1 : NOSP
        regionNeighbour = setdiff(unique(neighbours(SLIC_SEGMENT == i,:)),i);
        ADJMAT(i,regionNeighbour) = true;
    end
    % Throw exception if adjMat is not symmetric.
    msg = 'AdjM is not symmetric!!';
    assert(isequal(ADJMAT, ADJMAT'), msg);
    ADJ_TIME = toc(ADJ_TIME);
end

%% Compute Superpixel Features and Labels
function ComputeSPFeatures
    global IMG LABEL GRAY_IMG SLIC_SEGMENT NOSP ...
        SLIC_MASK FEATURE_VECTOR LABEL_VECTOR FEATURE_TIME ...
        REGIONPROP HEIGHT WIDTH;
    FEATURE_TIME = tic;
    FeatureDim = 134;
    voting = 0.5;  % To vote for superpixel label, if pixels > voting then labeled as floor
    % construct zeros vators for labels and feature vector
    LABEL_VECTOR = zeros(1,NOSP,'int8');
    FEATURE_VECTOR = zeros(FeatureDim,NOSP);
    % seperate the r-g-b intensity vectors from image
    red = IMG(:, :, 1);
    green = IMG(:, :, 2);
    blue = IMG(:, :, 3);
    hsv = rgb2hsv(IMG);
    hue = hsv(:,:,1);
    sat = hsv(:,:,2);
    gray = hsv(:,:,3);
    GRAY_IMG = rgb2gray(IMG);
    
    % texture filter
    lmfilter = makeLMfilters; % LM texture filters
    ntf = size(lmfilter, 3);  % # of texture filter
    lmtexture = struct('Response', {}); % LM texture filter response
    for i = 1:ntf
        lmtexture(i,1).Response = abs(imfilter(im2single(gray), lmfilter(:, :, i), 'same'));  
    end
    imtextures = cat(3,lmtexture.Response);
    [texth, texthist] = max(imtextures, [], 3);
    % region stats
    REGIONPROP  = regionprops(SLIC_SEGMENT,'Area', 'Centroid','Eccentricity','PixelList','PixelIdxList');
    FEATURE_TIME = toc(FEATURE_TIME);
    %% Extract features from each super pixel in image
    msg = 'Extract features from each super pixel in image';
    fprintf('\n%s',msg);
    fprintf('\nTotal Super Pixel in image : %d',NOSP);
    spftime = tic;
    for i = 1:NOSP
        fv = 0;
        pix = REGIONPROP(i,1).PixelIdxList;  % pixel indeces labelled as i
        % assign label Floor/Non-Floor if greater than voting
        reglabels = LABEL(pix);
        lval = size(reglabels(reglabels == 1),1)/size(reglabels,1);
        if (lval > voting)
            LABEL_VECTOR(1,i) = 1;   % Positive Class for SVM
        else
            LABEL_VECTOR(1,i) = -1;  %Negative Class for SVM
        end
        % label mask for overlay
        SLIC_MASK(pix) = lval > voting;
        
        % color features
        r = mean(red(pix));
        g = mean(green(pix));
        b = mean(blue(pix));
        rgb = [r g b];
        rgb(isnan(rgb)) = 0;
        hsv = rgb2hsv(rgb);
        
        FEATURE_VECTOR(fv+[1:6],i) = [rgb hsv];
        fv = fv + 6;
        % hue histogram 16 bin
        FEATURE_VECTOR(fv+[1:16],i) = hist(hue(pix),16)/sum(hist(hue(pix),16));
        fv = fv + 16;
        % saturation histogram 4 bin
        FEATURE_VECTOR(fv+[1:4],i) = hist(sat(pix),4)/sum(hist(sat(pix),4));
        fv = fv + 4;
        % value histogram 4 bin
        FEATURE_VECTOR(fv+[1:4],i) = hist(gray(pix),4)/sum(hist(gray(pix),4));
        fv = fv + 4;
        
        % locations features
        FEATURE_VECTOR(fv+1,i) = REGIONPROP(i,1).Centroid(1)/WIDTH;  % normalized x
        FEATURE_VECTOR(fv+2,i) = REGIONPROP(i,1).Centroid(2)/HEIGHT;  % normalized y
        xypercentile = prctile(REGIONPROP(i,1).PixelList,[10,90]); % 10,90 percentile of x,y
        FEATURE_VECTOR(fv+[3:4],i) = xypercentile(:,1)/WIDTH; %normalized x 10,90 percentile
        FEATURE_VECTOR(fv+[5:6],i) = xypercentile(:,2)/HEIGHT; %normalized y 10,90 percentile
        fv = fv + 6;
        
        % shape, eccentricity features
        FEATURE_VECTOR(fv+1,i) = REGIONPROP(i,1).Area/WIDTH/HEIGHT; % no of pixels in superpixel / Total pixel
        FEATURE_VECTOR(fv+2,i) = REGIONPROP(i,1).Eccentricity;
        fv = fv + 2;
        % mean texture features for each super pixel
        for k = 1:ntf
            FEATURE_VECTOR(fv+k,i) = mean(lmtexture(k,1).Response(pix));
        end
        fv = fv + ntf;
        
        % histogram of maximum response texture filter 
        FEATURE_VECTOR(fv+(1:ntf),i) = hist(texthist(pix), (1:ntf))+1;
        FEATURE_VECTOR(fv+(1:ntf),i) = FEATURE_VECTOR(fv+(1:ntf),i) ./ sum(FEATURE_VECTOR(fv+(1:ntf),i));
        % if any value is nan replace it by 0
        FEATURE_VECTOR((isnan(FEATURE_VECTOR(:,i))),i) = 0;
    end
    spftime = toc(spftime);
    FEATURE_TIME = FEATURE_TIME+spftime;
end

%% Compute Position Density Map
function ComputePositionDensity
    global LABEL POS_DENSITY_IMAGE POS_DENSITY_TIME HEIGHT WIDTH X Y bandwidth;
    POS_DENSITY_TIME  = tic;
    [X1,X2] = meshgrid(1:HEIGHT,1:WIDTH);
    XI = [X1(:) X2(:)];                                                                                                                        
    floorpix = find(LABEL==1);
    Data = XI(floorpix,:);
    [bandwidth,POS_DENSITY_IMAGE,X,Y]=kde2d(Data,HEIGHT,[0,0],[WIDTH,HEIGHT]);
    fprintf('\nBandwidth of KDE : %f',bandwidth);
    POS_DENSITY_IMAGE = POS_DENSITY_IMAGE + abs(min(POS_DENSITY_IMAGE(:)));
    POS_DENSITY_IMAGE = POS_DENSITY_IMAGE / max(POS_DENSITY_IMAGE(:));
    POS_DENSITY_TIME = toc(POS_DENSITY_TIME);
end

%% Compute FloorBoundary Map
function ComputeFloorBoundary
    global GRAY_IMG HEIGHT WIDTH FLOOR_BOUNDARY_MAP THRESHOLD FLOOR_BOUNDARY_TIME VL HL IL FPolyX FPolyY ...
        PolyX PolyY lines;
    lines = []; VL = []; HL = []; IL=[]; FPolyX=[];FPolyY=[]; PolyX=[]; PolyY=[];    
    FLOOR_BOUNDARY_TIME = tic;
    lines = getLargeEdges(GRAY_IMG,2);
    % convert all line theta to first quadrant
    angle  = radtodeg(lines(:,5));
    angle(angle<0) =  angle(angle<0)+180;
    angle(angle>90) =  180-angle(angle>90);
    % Dividing lines in 3 categories 85-90 Vetical Lines,0-10
    % Horizontal Lines 20-65 Inclined Lines
    VL=find(angle>=85 & lines(:,4)>=HEIGHT/2 & lines(:,7)>=60);
    HL=find(angle<=10 & lines(:,4)>=HEIGHT/2 & lines(:,7)>=30);
    IL=find(angle>=20 & angle<=65 & lines(:,4)>=HEIGHT/2 & lines(:,7)>=30);
    %construct Polygon from vertical end points
    Min_x = min(min(lines(:,[1 2])));
    Max_x = max(max(lines(:,[1 2])));
    VLEndPoints = [Min_x HEIGHT;lines(VL, [2 4])];
    VLEndPoints = sortrows(VLEndPoints,[1 -2]);
    VLEndPoints(end+1,:) = [Max_x HEIGHT];
    Length = numel(VLEndPoints(:,1));
    len = 10;
    while Length < 4 && len<=50
        clear VLEndPoints;
        VL=find(angle>=85 & lines(:,4)>=HEIGHT/2 & lines(:,7)>= (60-len));
        VLEndPoints = [Min_x HEIGHT;lines(VL, [2 4])];
        VLEndPoints = sortrows(VLEndPoints,[1 -2]);
        VLEndPoints(end+1,:) = [Max_x HEIGHT];
        Length = numel(VLEndPoints(:,1));
        len = len + 5;
    end
    PolyX = VLEndPoints(:,1);
    PolyY = VLEndPoints(:,2);
    AllLines = lines([HL;IL],:);
    BS = BottomScore(AllLines,PolyX,PolyY);
    % New Hull from Horizontal Lines selected
    AL_X = AllLines(BS>THRESHOLD, [1 2]);
    AL_Y = AllLines(BS>THRESHOLD, [3 4]);
    Points = [[PolyX(1) PolyX(1) PolyY(1) PolyY(1)];[PolyX(end) PolyX(end) PolyY(end) PolyY(end)];[AL_X AL_Y]];
    Points = sortrows(Points,[1 2 3 4]);
    AL_X =Points(:,[1 2]);
    AL_Y =Points(:,[3 4]);
    [AL_X,IND] = sort(AL_X,2);
    pl = length(IND(:,1));
    for pi = 1:pl
        AL_Y(pi,:) = permute(AL_Y(pi,:),IND(pi,:));
    end
    AL_X = AL_X';
    AL_Y = AL_Y';
    AL_X = AL_X(:);
    AL_Y = AL_Y(:);
    Points = [AL_X AL_Y];
    hlind_x = diff(Points(:,1),[],1);
    hlind_x = [1;hlind_x];
    hlind_x = hlind_x > 0;
    Points = Points(hlind_x,:);
    Points(end+1,:) = [PolyX(end) PolyY(end)];
    Points(end+1,:) = [PolyX(1) PolyY(1)];
    FPolyX = Points(:,1);
    FPolyY = Points(:,2);
    %Polygon Mask
    FLOOR_BOUNDARY_MAP = poly2mask(FPolyX,FPolyY,WIDTH,HEIGHT);
    FLOOR_BOUNDARY_TIME = toc(FLOOR_BOUNDARY_TIME);
end
