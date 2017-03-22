function ResultAnalysis(InputDir,OutputDir)
    addpaths;
    time = tic;
    fnName = 'ResultAnalysis:';
    msg = '----------begins----------';
    fprintf('\n%s %s',fnName,msg);
    if ~exist(InputDir,'dir')
       error('\n%s',msg);
    else
        INPUTDIR = InputDir;
        OutputImages = dir([InputDir '*' '.mat']);
    end
    % Create Output Dir
    if ~exist(OutputDir,'dir')
        mkdir(OutputDir);
    end
    
    ResultAnalysis = [OutputDir '/ResultAnalysis'];
    fp=fopen(ResultAnalysis,'a+');
    fprintf(fp,'ID,Name,SVM_Accuracy,SVM_Recall,SVM_Precision,SVM_Gmean,SVM_Specificity,SVM_tp,SVM_fp,SVM_fn,SVM_tn,SVM_total,');
    fprintf(fp,'POSITION_Accuracy,POSITION_Recall,POSITION_Precision,POSITION_Gmean,POSITION_Specificity,POSITION_tp,POSITION_fp,POSITION_fn,POSITION_tn,POSITION_total,');
    fprintf(fp,'BOUNDARY_Accuracy,BOUNDARY_Recall,BOUNDARY_Precision,BOUNDARY_Gmean,BOUNDARY_Specificity,BOUNDARY_tp,BOUNDARY_fp,BOUNDARY_fn,BOUNDARY_tn,BOUNDARY_total,');
    fprintf(fp,'GENERIC_Accuracy,GENERIC_Recall,GENERIC_Precision,GENERIC_Gmean,GENERIC_Specificity,GENERIC_tp,GENERIC_fp,GENERIC_fn,GENERIC_tn,GENERIC_total,');
    fprintf(fp,'GRABCUT_Accuracy,GRABCUT_Recall,GRABCUT_Precision,GRABCUT_Gmean,GRABCUT_Specificity,GRABCUT_tp,GRABCUT_fp,GRABCUT_fn,GRABCUT_tn,GRABCUT_total,');
    fprintf(fp,'GRAB_PIXEL_Accuracy,GRAB_PIXEL_Recall,GRAB_PIXEL_Precision,GRAB_PIXEL_Gmean,GRAB_PIXEL_Specificity,GRAB_PIXEL_tp,GRAB_PIXEL_fp,GRAB_PIXEL_fn,GRAB_PIXEL_tn,GRAB_PIXEL_total\n');
    fclose(fp);


    OnlyAccGmean = [OutputDir '/GmeanAcc'];
    fp=fopen(OnlyAccGmean,'a+');
    fprintf(fp,'ID,Name,SVM_Accuracy,SVM_Gmean,');
    fprintf(fp,'POSITION_Accuracy,POSITION_Gmean,');
    fprintf(fp,'BOUNDARY_Accuracy,BOUNDARY_Gmean,');
    fprintf(fp,'GENERIC_Accuracy,GENERIC_Gmean,');
    fprintf(fp,'GRABCUT_Accuracy,GRABCUT_Gmean,');
    fprintf(fp,'GRAB_PIXEL_Accuracy,GRAB_PIXEL_Gmean\n');
    fclose(fp);
    
    TimeAnalysis = [OutputDir '/TimeAnalysis'];  
    fp=fopen(TimeAnalysis,'a+');
    fprintf(fp,'ID,Name,Height,Width,Thresh,SLIC_TIME,ADJ_TIME,FEATURE_TIME,POS_DENSITY_TIME,FLOOR_BOUNDARY_TIME,SVM_LEARNING_TIME,SVM_PREDICT_TIME,POSITION_PREDICT_TIME,GENERIC_PREDICT_TIME,BOUNDARY_PREDICT_TIME, GMM_TIME, GRAB_PIXEL_TIME\n');
    fclose(fp);
    N = size(OutputImages,1);    % Total Tested Images Mat files
    fprintf('\nTotal MAT loaded : %d\n',N);
    for ID = 1:N
        fprintf('\n---%d %s----',ID,OutputImages(ID).name);
        load([InputDir OutputImages(ID).name]);
        % Evaluate
        SVM = Evaluate(LABEL_VECTOR,SVM_LABEL);
        POSITION = Evaluate(LABEL_VECTOR,POSITION_LABEL);
        GENERIC = Evaluate(LABEL_VECTOR,GENERIC_LABEL);
        GRABCUT = Evaluate(LABEL_VECTOR,GRABCUT_LABELS);
        BOUNDARY = Evaluate(LABEL_VECTOR,BOUNDARY_LABEL);
        GLABEL = double(LABEL);
        GLABEL(GLABEL==0) = -1;
        GRABCUT_PIXEL = Evaluate(GLABEL,GRABCUTPIXEL_TRIMAP);   
        
        %save Predicted Mask
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_GT.jpg'];
        SavePredictedMask(IMG,SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,LABEL_VECTOR,NAME);
        
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_SVM.jpg'];
        SavePredictedMask(IMG,SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,SVM_LABEL,NAME);
        
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_POSITION.jpg'];
        SavePredictedMask(IMG,SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,POSITION_LABEL,NAME);
        
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_GENERIC.jpg'];
        SavePredictedMask(IMG,SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,GENERIC_LABEL,NAME);
        
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_GRABCUT.jpg'];
        SavePredictedMask(IMG,SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,GRABCUT_LABELS,NAME);
        
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_BOUNDARY.jpg'];
        SavePredictedMask(IMG,SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,BOUNDARY_LABEL,NAME);
        
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_FLOORBOUNDARY.jpg'];
        BLENDIMG = imoverlay(IMG,FLOOR_BOUNDARY_MAP==1,[0 1 0]);
        BLENDIMG = imfuse(BLENDIMG,IMG,'blend','Scaling','joint');
        imwrite(BLENDIMG,NAME);
        
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_GRABCUT_PIXEL.jpg'];
        BLENDIMG = imoverlay(IMG,GRABCUTPIXEL_TRIMAP==1,[0 1 0]);
        BLENDIMG = imfuse(BLENDIMG,IMG,'blend','Scaling','joint');
        imwrite(BLENDIMG,NAME);
        
        MASK = zeros(size(SLIC_SEGMENT));
        for i= 1:NOSP
            MASK(SLIC_SEGMENT == i) = GRABCUT_LABELS(i);
        end
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_FINAL.jpg'];
        BLENDIMG = imoverlay(IMG,MASK==1,[0 1 0]);
        BLENDIMG = imfuse(BLENDIMG,IMG,'blend','Scaling','joint');
        imwrite(BLENDIMG,NAME);
        
        % PLOT DENSITY Image
        NAME = [OutputDir OutputImages(ID).name(1:end-4) '_PDI.jpg'];
        PlotDensity(POS_DENSITY_IMAGE,X,Y,NAME,WIDTH,HEIGHT,1);
        
        
%         %Plot Boundaries
%         f=figure;
%         set(f, 'visible', 'off');
%         imagesc(IMG);
%         colormap(gray);
%         hold on;
%         plot(lines(VL, [1 2])', lines(VL, [3 4])','b');
%         plot(lines(HL, [1 2])', lines(HL, [3 4])','g');
%         plot(lines(IL, [1 2])', lines(IL, [3 4])','r');
%         plot(PolyX,PolyY,'y');
%         plot(FPolyX,FPolyY,'c');
%         saveas(f,[OutputDir OutputImages(ID).name(1:end-4) '_LINES.jpg']);        
        
        %Result Analysis
        fp=fopen(ResultAnalysis,'a+');
        fprintf(fp,'%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,',...
            ID,OutputImages(ID).name,SVM.Accuracy,SVM.Recall,SVM.Precision,SVM.Gmean,SVM.Specificity,SVM.tp,SVM.fp,SVM.fn,SVM.tn,SVM.total);
        fprintf(fp,'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,',...
            POSITION.Accuracy,POSITION.Recall,POSITION.Precision,POSITION.Gmean,POSITION.Specificity,POSITION.tp,POSITION.fp,POSITION.fn,POSITION.tn,POSITION.total);
        fprintf(fp,'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,',...
            BOUNDARY.Accuracy,BOUNDARY.Recall,BOUNDARY.Precision,BOUNDARY.Gmean,BOUNDARY.Specificity,BOUNDARY.tp,BOUNDARY.fp,BOUNDARY.fn,BOUNDARY.tn,BOUNDARY.total);
        fprintf(fp,'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,',...
            GENERIC.Accuracy,GENERIC.Recall,GENERIC.Precision,GENERIC.Gmean,GENERIC.Specificity,GENERIC.tp,GENERIC.fp,GENERIC.fn,GENERIC.tn,GENERIC.total);
        fprintf(fp,'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,',...
            GRABCUT.Accuracy,GRABCUT.Recall,GRABCUT.Precision,GRABCUT.Gmean,GRABCUT.Specificity,GRABCUT.tp,GRABCUT.fp,GRABCUT.fn,GRABCUT.tn,GRABCUT.total);
        fprintf(fp,'%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n',...
            GRABCUT_PIXEL.Accuracy,GRABCUT_PIXEL.Recall,GRABCUT_PIXEL.Precision,GRABCUT_PIXEL.Gmean,GRABCUT_PIXEL.Specificity,GRABCUT_PIXEL.tp,GRABCUT_PIXEL.fp,GRABCUT_PIXEL.fn,GRABCUT_PIXEL.tn,GRABCUT_PIXEL.total);
        
        fclose(fp);
        
        OnlyAccGmean = [OutputDir '/GmeanAcc'];
        fp=fopen(OnlyAccGmean,'a+');
        fprintf(fp,'%d,%s,%f,%f,',...
            ID,OutputImages(ID).name,SVM.Accuracy,SVM.Gmean);
        fprintf(fp,'%f,%f,',...
            POSITION.Accuracy,POSITION.Gmean);
        fprintf(fp,'%f,%f,',...
            BOUNDARY.Accuracy,BOUNDARY.Gmean);
        fprintf(fp,'%f,%f,',...
            GENERIC.Accuracy,GENERIC.Gmean);
        fprintf(fp,'%f,%f,',...
            GRABCUT.Accuracy,GRABCUT.Gmean);
        fprintf(fp,'%f,%f\n',...
            GRABCUT_PIXEL.Accuracy,GRABCUT_PIXEL.Gmean);
        
        fclose(fp);


        %Time Analysis
        TimeAnalysis = [OutputDir '/TimeAnalysis'];  
        fp=fopen(TimeAnalysis,'a+');
        fprintf(fp,'%d,%s,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n',...
            ID,OutputImages(ID).name,HEIGHT,WIDTH,THRESHOLD,SLIC_TIME,ADJ_TIME,FEATURE_TIME,POS_DENSITY_TIME,FLOOR_BOUNDARY_TIME,SVM_LEARNING_TIME,SVM_PREDICT_TIME,BOUNDARY_PREDICT_TIME,GENERIC_PREDICT_TIME,BOUNDARY_PREDICT_TIME,GMM_TIME,GRAB_PIXEL_TIME);       
        fclose(fp);
    end
    % PLOT DENSITY Image
    NAME = [OutputDir 'PositionDensityMap.jpg'];
    PlotDensity(POSITION_DENSITY_MAP,X,Y,NAME,WIDTH,HEIGHT,K);
    toc(time);
end

function Metrics = Evaluate(LABEL_VECTOR,PREDICTED_LABEL)
    Metrics = struct('Accuracy', {}, 'Recall', {}, 'Precision', {},'Gmean',{},'Specificity',{},...
        'tp',{},'fp',{},'fn',{},'tn',{},'total',{});
    % True positive,false positive,false negative and true negative
    tp_idx = intersect(find(LABEL_VECTOR == 1),find(PREDICTED_LABEL == 1));
    fp_idx = intersect(find(LABEL_VECTOR == -1),find(PREDICTED_LABEL == 1));
    fn_idx = intersect(find(LABEL_VECTOR == 1),find(PREDICTED_LABEL == -1));
    tn_idx = intersect(find(LABEL_VECTOR == -1),find(PREDICTED_LABEL == -1));
    
    Metrics(1).tp = numel(tp_idx);
    Metrics(1).fp = numel(fp_idx);
    Metrics(1).fn = numel(fn_idx);
    Metrics(1).tn = numel(tn_idx);
    Metrics(1).total = Metrics(1).tp + Metrics(1).fp + Metrics(1).fn + Metrics(1).tn;
    %metrics
    Metrics(1).Accuracy = (Metrics(1).tp + Metrics(1).tn) / (Metrics(1).tp + Metrics(1).fp + Metrics(1).fn + Metrics(1).tn);
    Metrics(1).Precision = Metrics(1).tp / (Metrics(1).tp + Metrics(1).fp);
    Metrics(1).Recall = Metrics(1).tp / (Metrics(1).tp + Metrics(1).fn);
    Metrics(1).Specificity = Metrics(1).tn/(Metrics(1).fp+Metrics(1).tn);
    Metrics(1).Gmean = sqrt(Metrics(1).Specificity*Metrics(1).Recall);
end

function SavePredictedMask(IMG,SLIC_SEGMENT,SLIC_BOUNDARIES,NOSP,TRIMAP,NAME)    
    MASK = zeros(size(SLIC_SEGMENT));
    for i= 1:NOSP
        MASK(SLIC_SEGMENT == i) = TRIMAP(i);
    end
    %BLENDIMG = imoverlay(IMG,MASK==-1,[0 1 0]);
    BLENDIMG = imoverlay(IMG,MASK==1,[0 1 0]);
    BLENDIMG = imoverlay(BLENDIMG,MASK==0,[0 0 1]);
    BLENDIMG = imoverlay(BLENDIMG,MASK==2,[1 1 0]);
    BLENDIMG = imfuse(BLENDIMG,IMG,'blend','Scaling','joint');
    BLENDIMG = imoverlay(BLENDIMG,SLIC_BOUNDARIES,[1 1 1]);
    imwrite(BLENDIMG,NAME);
end

function PlotDensity(MAP,X,Y,NAME,WIDTH,HEIGHT,K)
    % plot the data and the density estimate
    f = figure('visible','off');    
    contour3(X,Y,MAP,50), hold on
    plot(X,Y,'r.','MarkerSize',50);
    xlabel(['Image Width:' int2str(WIDTH)]); ylabel(['Image Height:' int2str(HEIGHT)]);
    zlabel('Probability Density');
    title(['Parzen-Window Density Map Estimate of Floor Region on ' int2str(K) ' images of resolution ' int2str(HEIGHT) ' x ' int2str(WIDTH)]);
    fprintf('...SaveDensityFig ');
    set(f, 'paperpositionmode', 'auto');
    saveas(f, NAME);
end
