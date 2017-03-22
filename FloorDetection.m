% Get Mat containing SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,REGIONSIZE,ADJMAT,FEATURE_VECTOR,LABEL_VECTOR,IMG,LABEL_MASK  
% USAGE: FloorDetection(InputDir,OutputDir,RegionSize)
%
% INPUT:
%    InputDir  : Computed Image Directory
%    OutputDir : OutputDirectory to result after each step
%    Confidence : Confidence for passing to floor
% OUTPUT:
%       Append Detected floor mask after each process to input file
% Author:Sanchit Aggarwal
% Date:17-December-2013 10:29 A.M.

function FloorDetection(InputDir,OutputDir,Confidence,w)
    global ALLIDX ComputedImages K N INPUTDIR SVM_LEARNING_TIME POSITION_DENSITY_MAP Model CONFIDENCE ...
        SVM_SCORES SVM_PREDICT_TIME SVM_LABEL SVM_TRIMAP W ...
        POSITION_SCORES POSITION_PREDICT_TIME POSITION_LABEL POSITION_TRIMAP ...
        GENERIC_SCORES GENERIC_PREDICT_TIME GENERIC_LABEL GENERIC_TRIMAP ...
        BOUNDARY_SCORES BOUNDARY_DIST BOUNDARY_PREDICT_TIME FLOOR_BOUNDARYPROP ...
        GMM_TIME GRABCUT_LABELS Iteration K  BOUNDARY_LABEL BOUNDARY_TRIMAP GRABCUTPIXEL_TRIMAP GRAB_PIXEL_TIME;
    addpaths;
    time = tic;
    fnName = 'FloorDetection:';
    msg = '----------begins----------';
    fprintf('\n%s %s',fnName,msg);
    if ~exist(InputDir,'dir')
       error('\n%s',msg);
    else
        INPUTDIR = InputDir;
        ComputedImages = dir([InputDir '*' '.mat']);
        CONFIDENCE = Confidence/100;
        W=w;
    end
    % Create Output Dir
    if ~exist(OutputDir,'dir')
        mkdir(OutputDir);
    end
    N = size(ComputedImages,1);    % Total Training Images   
    RATIO = 0.7;
    K = floor(N*RATIO);
    fprintf('\nTotal MAT loaded : %d\n',N);
    randn('state',1) ;
    rand('state',1) ;
    vl_twister('state',1) ;
    %ALLIDX = randperm(N);  %original setup date 28 april 2014
    ALLIDX = 1:N; %for producing failure results
    %Compute SVM Classifier Model
    ComputeGenericCues;
    % Detecting Floors on Testing Images 
    for ID = K+1:N
        fprintf('\n---%d %s----',ID,ComputedImages(ALLIDX(ID)).name);
        load([InputDir ComputedImages(ALLIDX(ID)).name]);
        % Predict SVM Scores and Labels
        PredictSVM(FEATURE_VECTOR,NOSP);
        % Predict Position Scores and Labels
        PredictPosition(REGIONPROP,NOSP);
        % Predict Floor Boundary Scores
        PredictBoundaryCues(REGIONPROP,FLOOR_BOUNDARY_MAP,NOSP);
        % PredictGeneric Scores and Labels
        PredictGeneric(NOSP);
        % Pedict GrabCut (Specifc Appearance Cues)
        PredictGrabCut(FEATURE_VECTOR([1,2,3,31,32],:),ADJMAT,NOSP);
        % Predict Grabcut at pixel Level
        GrabCutPixel(IMG,SLIC_SEGMENT,NOSP);
        
        save([OutputDir ComputedImages(ALLIDX(ID)).name],'IMG','LABEL',...
            'GRAY_IMG','SLIC_TIME','SLIC_SEGMENT','NOSP','SLIC_BOUNDARIES', ...
        'SLIC_MASK','ADJ_TIME','ADJMAT','FEATURE_VECTOR','LABEL_VECTOR','FEATURE_TIME', ...
        'POS_DENSITY_IMAGE','POS_DENSITY_TIME','HEIGHT','WIDTH','X','Y',...
        'FLOOR_BOUNDARY_MAP','THRESHOLD','FLOOR_BOUNDARY_TIME','REGIONPROP',...
        'SVM_LEARNING_TIME','POSITION_DENSITY_MAP','Model','CONFIDENCE', ...
        'SVM_SCORES','SVM_PREDICT_TIME','SVM_LABEL','SVM_TRIMAP', ...
        'POSITION_SCORES','POSITION_PREDICT_TIME','POSITION_LABEL','POSITION_TRIMAP', ...
        'GENERIC_SCORES','GENERIC_PREDICT_TIME','GENERIC_LABEL','GENERIC_TRIMAP', ...
        'BOUNDARY_SCORES','BOUNDARY_DIST','BOUNDARY_PREDICT_TIME','FLOOR_BOUNDARYPROP',...
        'GMM_TIME','GRABCUT_LABELS','Iteration','K','BOUNDARY_LABEL','BOUNDARY_TRIMAP','GRABCUTPIXEL_TRIMAP','GRAB_PIXEL_TIME');
    end
    toc(time);
end

%%Compute SVM Classifier Model
function ComputeGenericCues
    global ALLIDX ComputedImages K N INPUTDIR SVM_LEARNING_TIME POSITION_DENSITY_MAP Model;
    FeatureVector = [];
    LabelVector = [];
    POSITION_DENSITY_MAP = 0;
    for ID = 1:K
        load([INPUTDIR ComputedImages(ALLIDX(ID)).name]);
        FeatureVector = cat(2,FeatureVector,FEATURE_VECTOR);
        LabelVector = cat(2,LabelVector,LABEL_VECTOR);
        POSITION_DENSITY_MAP = POSITION_DENSITY_MAP+POS_DENSITY_IMAGE;
    end
    TestFeatureVector = [];
    TestLabelVector = [];
    for ID = K+1:N
        load([INPUTDIR ComputedImages(ALLIDX(ID)).name]);
        TestFeatureVector = cat(2,TestFeatureVector,FEATURE_VECTOR);
        TestLabelVector = cat(2,TestLabelVector,LABEL_VECTOR);
    end
    POSITION_DENSITY_MAP = POSITION_DENSITY_MAP./K;
    SVM_LEARNING_TIME = tic;
    %% Learn SVM model
    randn('state',1) ;
    rand('state',1) ;
    vl_twister('state',1) ; 
    Model = struct('w', {},'A',{},'B',{},'StartingIter',{});
    conf.C = 10^6 ;
    conf.biasMultiplier = 1 ;
    % Compute feature map
    psix=vl_homkermap(FeatureVector,1,'kchi2','gamma',.5);
    Dimension = size(psix,1);
    Model(1).w = zeros(Dimension+1,1) ;
    Model(1).StartingIter = 1;
    Model(1).A = 0;
    Model(1).B = 0;
    prec = ones(Dimension+1,1) ; 
    prec(end) = .1/conf.biasMultiplier ;
    LabelVector = int8(LabelVector);
    conf.numiter = length(LabelVector) ;
    lambda = 1 / (conf.C *  conf.numiter) ;
    Model.w = vl_pegasos(psix,LabelVector,lambda,  ...
            	   'startingModel', Model.w, ...
                   'startingIteration', Model.StartingIter, ...
             	   'numIterations', conf.numiter, ...
             	   'biasMultiplier', conf.biasMultiplier, ...
             	   'preconditioner', prec);
    % New Starting Iter
    Model.StartingIter = Model.StartingIter + conf.numiter;
    % sigmoid parameters for probabilistic scores
    psix = vl_homkermap(TestFeatureVector, 1, 'kchi2', 'gamma', .5);
    out = Model.w(1:end-1)' * psix + Model.w(end)' * ones(1,size(psix,2));
    target = TestLabelVector;     
    [Model.A,Model.B] = getSigmoidParam(out,target,Model.A,Model.B);
    SVM_LEARNING_TIME = toc(SVM_LEARNING_TIME);
end

%%Predict SVM Scores and Labels
function PredictSVM(FEATURE_VECTOR,NOSP)
    global Model CONFIDENCE SVM_SCORES SVM_PREDICT_TIME SVM_LABEL SVM_TRIMAP;
    SVM_PREDICT_TIME = tic;
    psix= vl_homkermap(FEATURE_VECTOR, 1, 'kchi2', 'gamma', .5);
    SCORES = Model.w(1:end-1)' * psix + Model.w(end)' * ones(1,size(psix,2));
    SVM_PREDICT_TIME = toc(SVM_PREDICT_TIME);
    % assgn +1 label for postitive scores and -1 label for Negative scores
    SVM_LABEL = zeros(1,NOSP,'int8');
    SVM_LABEL(SCORES < 0) = -1;  
    SVM_LABEL(SCORES >= 0) = 1;
    % Probability estimate for +1 class
    SVM_SCORES = zeros(NOSP,2);
    SVM_SCORES(:,1) = 1 - (1 ./ (1+exp(Model.A.*SCORES + Model.B))); %Probabilities for -ve sample (Non-Floor)
    SVM_SCORES(:,2) = 1 ./ (1+exp(Model.A.*SCORES + Model.B)); %Probabilities for +ve sample (Floor)
    %SVM Trimap for grabcut
    SVM_TRIMAP = zeros(1,NOSP,'int8');
    SVM_TRIMAP(SCORES < 0) = 0;
    SVM_TRIMAP(SCORES >= 0) = 2;
    SVM_TRIMAP(SVM_SCORES(:,2) >= CONFIDENCE) = 1;
    SVM_TRIMAP(SVM_SCORES(:,1) >= CONFIDENCE) = -1;
end

%%Predict Position Scores and Labels
function PredictPosition(REGIONPROP,NOSP)
    global POSITION_DENSITY_MAP CONFIDENCE POSITION_SCORES ...
        POSITION_PREDICT_TIME POSITION_LABEL POSITION_TRIMAP;
    POSITION_PREDICT_TIME = tic;
    POSITION_SCORES = zeros(NOSP,2);
    POSITION_LABEL = zeros(1,NOSP,'int8');
    for i = 1:NOSP
        pix = REGIONPROP(i,1).PixelIdxList;  % pixel indeces labelled as i
        val = mean(POSITION_DENSITY_MAP(pix));
        POSITION_SCORES(i,2) = val;
        POSITION_SCORES(i,1) = 1-val;
        
        % assign label Floor/Non-Floor
        if (val >= 0.5)
            POSITION_LABEL(1,i) = 1;   % Positive Class/Floor Positions
        else
            POSITION_LABEL(1,i) = -1;  %Negative Class/Non-Floor Positions
        end
    end
    POSITION_PREDICT_TIME = toc(POSITION_PREDICT_TIME);
    POSITION_TRIMAP = zeros(1,NOSP,'int8');
    POSITION_TRIMAP(POSITION_LABEL == -1) = 0;
    POSITION_TRIMAP(POSITION_LABEL == 1) = 2;
    POSITION_TRIMAP(POSITION_SCORES(:,2) >= CONFIDENCE) = 1;
    POSITION_TRIMAP(POSITION_SCORES(:,1) >= CONFIDENCE) = -1;
end

% PredictGeneric Scores and Labels
function PredictGeneric(NOSP)
    global CONFIDENCE SVM_SCORES POSITION_SCORES GENERIC_SCORES W...
        GENERIC_PREDICT_TIME GENERIC_LABEL GENERIC_TRIMAP BOUNDARY_SCORES;
    W = W./sum(W);
    
    GENERIC_PREDICT_TIME = tic;
    GENERIC_SCORES = zeros(NOSP,2);
    GENERIC_LABEL = zeros(1,NOSP,'int8');
    GENERIC_SCORES = W(1)*SVM_SCORES + W(2)*POSITION_SCORES + W(3)*BOUNDARY_SCORES;
    GENERIC_LABEL(GENERIC_SCORES(:,2) < 0.5) = -1;  
    GENERIC_LABEL(GENERIC_SCORES(:,2) >= 0.5) = 1;
    GENERIC_PREDICT_TIME = toc(GENERIC_PREDICT_TIME);
    GENERIC_TRIMAP = zeros(1,NOSP,'int8');
    GENERIC_TRIMAP(GENERIC_LABEL == -1) = 0;
    GENERIC_TRIMAP(GENERIC_LABEL == 1) = 2;
    GENERIC_TRIMAP(GENERIC_SCORES(:,2) >= CONFIDENCE) = 1;
    GENERIC_TRIMAP(GENERIC_SCORES(:,1) >= CONFIDENCE) = -1;
end

% Predict Floor Boundary Scores
function PredictBoundaryCues(REGIONPROP,FLOOR_BOUNDARY_MAP,NOSP)
    global CONFIDENCE BOUNDARY_SCORES BOUNDARY_DIST BOUNDARY_PREDICT_TIME FLOOR_BOUNDARYPROP BOUNDARY_LABEL BOUNDARY_TRIMAP;
    BOUNDARY_PREDICT_TIME = tic;
    BOUNDARY_DIST = zeros(NOSP,1);
    BOUNDARY_SCORES = zeros(NOSP,2);
    FLOOR_BOUNDARYPROP  = regionprops(FLOOR_BOUNDARY_MAP,'Area', 'Centroid','PixelIdxList');
    for i = 1:NOSP
        pix = REGIONPROP(i,1).PixelIdxList;  % pixel indeces labelled as i
        regboundary = FLOOR_BOUNDARY_MAP(pix);
        lval = max(1/size(regboundary,1),(size(regboundary(regboundary == 1),1)-1)/size(regboundary,1));
        BOUNDARY_SCORES(i,2) = lval;  
        %BOUNDARY_DIST(i,1) = norm(REGIONPROP(i,1).Centroid-mean(FLOOR_BOUNDARYPROP.Centroid));
    end
    %BOUNDARY_DIST(:,1) = 1 - (BOUNDARY_DIST(:,1)-1)/max(BOUNDARY_DIST(:,1));
    %BOUNDARY_SCORES(:,2) = BOUNDARY_SCORES(:,2) .* BOUNDARY_DIST(:,1);
    BOUNDARY_SCORES(:,1) = 1-BOUNDARY_SCORES(:,2);
    BOUNDARY_PREDICT_TIME = toc(BOUNDARY_PREDICT_TIME);
    BOUNDARY_LABEL = zeros(1,NOSP,'int8');
    BOUNDARY_LABEL(BOUNDARY_SCORES(:,2) < 0.5) = -1;  
    BOUNDARY_LABEL(BOUNDARY_SCORES(:,2) >= 0.5) = 1;
    BOUNDARY_TRIMAP = zeros(1,NOSP,'int8');
    BOUNDARY_TRIMAP(BOUNDARY_LABEL == -1) = 0;
    BOUNDARY_TRIMAP(BOUNDARY_LABEL == 1) = 2;
    BOUNDARY_TRIMAP(BOUNDARY_LABEL(:,2) >= CONFIDENCE) = 1;
    BOUNDARY_TRIMAP(BOUNDARY_LABEL(:,1) >= CONFIDENCE) = -1;
    end

% Pedict GrabCut (Specifc Appearance Cues)
function PredictGrabCut(DATA,ADJMAT,NOSP)
    global GMM_TIME GENERIC_TRIMAP GRABCUT_LABELS BOUNDARY_SCORES Iteration;
    GMM_TIME = tic;
    K = 5;
    Gamma = 50;
    TotalIteration = 10;
    FGM = '';
    BGM = '';
    % Force Weight for Floor/Non-Floor sample
    L = computeL(Gamma);
    %Get N-Links weights
    Pairwise = ComputePairwise(DATA,ADJMAT,Gamma);
    Foreground = DATA(:,find(GENERIC_TRIMAP >= 1));
    Background = DATA(:,find(GENERIC_TRIMAP <= 0));
    if ~isstruct(FGM)
        [Mixture,FGM] = GaussianMixture(Foreground', 20, K, false, 1e5);
    end
    if ~isstruct(BGM)
        [Mixture,BGM] = GaussianMixture(Background', 20, K, false, 1e5);
    end
    GRABCUT_LABELS = zeros(1,NOSP,'int8');
    GRABCUT_LABELS = GRABCUT_LABELS';
    for Iteration = 1:TotalIteration
%         [Mixture,FGM] = GaussianMixture(Foreground', 20, K, false, 1e5);
%         [Mixture,BGM] = GaussianMixture(Background', 20, K, false, 1e5);
%         
        FGM = EMIterate(FGM, Foreground');
        BGM = EMIterate(BGM, Background');

        % Assign Unary Potential Matrix of size |V|x2 ,Unary(i,j) = the cost of assigning label j to node i.
        fprintf('\nAssign Unary Potential');
        N = size(DATA,2);
        Unary = sparse(N,2);
        % assign negative log-likelihood of foreground to background T-Links and vice-versa
        Unary(:,1) = GMClassLikelihood(BGM, DATA');% - log(BOUNDARY_SCORES(:,1));  % Non Floor T-Weight (Source)
        Unary(:,2) = GMClassLikelihood(FGM, DATA');% - log(BOUNDARY_SCORES(:,2));  % Floor T-Weight (Sink)
        % Assign weight based on initial trimap
        % Non -Floor weight
        Unary(find(GENERIC_TRIMAP==-1),1) = L;
        Unary(find(GENERIC_TRIMAP==-1),2) = 0;
%         %Floor weight
%         Unary(find(GENERIC_TRIMAP==1),1) = 0;
%         Unary(find(GENERIC_TRIMAP==1),2) = L;
        OldLabels = GRABCUT_LABELS;
        % Apply min-cut, max flow predicted labels will be 0 for Non-Floor and 1 for Floor
        [Flow,GRABCUT_LABELS] = maxflow(Pairwise,Unary);
        GRABCUT_LABELS = int8(GRABCUT_LABELS);
        Foreground = DATA(:,find(GRABCUT_LABELS == 1));
        Background = DATA(:,find(GRABCUT_LABELS == 0));
        
        COUNT = numel(find(OldLabels-GRABCUT_LABELS~=0));
        fprintf('\nIteration:%d Count:%d',Iteration,COUNT);
        if COUNT == 0
            break;
        end
    end
    GRABCUT_LABELS(GRABCUT_LABELS==0)=-1;
    GMM_TIME = toc(GMM_TIME);
end

function L = computeL(Gamma)
    L =  8*Gamma+1;
end

%% For NLinks weight of Graph Cut
function Pairwise = ComputePairwise(DATA,ADJMAT,Gamma)
    N = size(DATA,2);
    result = 0;
    edges = 0;
    Pairwise = sparse(N,N);
    Distance = sparse(N,N); % for distance between two neighbouring superpixel
    for i = 1:N
       for j = 1:N
          if ADJMAT(i,j)
              Pairwise(i,j) = sumsqr(DATA(:,i)-DATA(:,j));
              % assuming 4,5 is the centroid of superpixel
              Distance(i,j) = norm(DATA(4:5,i)-(DATA(4:5,i)));  
              result = result + Pairwise(i,j);
              edges = edges + 1;
          end
       end
    end
    Beta = 1/(2*result/edges);
    Pairwise = Gamma * exp(-Beta*Pairwise);
    Pairwise = Distance.*Pairwise;
end

function GrabCutPixel(IMG,SLIC_SEGMENT,NOSP)
    global GENERIC_TRIMAP GRABCUTPIXEL_TRIMAP GRAB_PIXEL_TIME;    
    GRAB_PIXEL_TIME = tic;
    TRIMAP = zeros(size(SLIC_SEGMENT));
    for i= 1:NOSP
        TRIMAP(SLIC_SEGMENT == i) = GENERIC_TRIMAP(i);
    end
    %{0:bg, 1:fg, 2:probably-bg, 3:probably-fg}
    TRIMAP(TRIMAP>0) = 3;
    TRIMAP(TRIMAP==0) = 2;
    TRIMAP(TRIMAP==-1) = 0;
    [GRABCUTPIXEL_TRIMAP,bgmodel,fgmodel] = cv.grabCut(im2uint8(IMG),uint8(TRIMAP),'MaxIter',10);
    %[GRABCUTPIXEL_TRIMAP,bgmodel,fgmodel] = cv.grabCut(im2uint8(IMG),GRABCUTPIXEL_TRIMAP,'MaxIter',1);
    GRABCUTPIXEL_TRIMAP = double(GRABCUTPIXEL_TRIMAP);
    GRABCUTPIXEL_TRIMAP(GRABCUTPIXEL_TRIMAP==0) = -1;
    GRABCUTPIXEL_TRIMAP(GRABCUTPIXEL_TRIMAP==2) = -1;
    GRABCUTPIXEL_TRIMAP(GRABCUTPIXEL_TRIMAP==3) = 1;
    GRAB_PIXEL_TIME = toc(GRAB_PIXEL_TIME);
end
