function Baseline(inputdir,ppmdir,superpixeldir,resultdir)
    global ID Name HEIGHT WIDTH PPM_TIME SUPERPIXEL_TIME PREDICTIONTIME boxlayout surface_labels MASK layoutdir;
    addpaths;
    HEIGHT=512;
    WIDTH=512;
    InputMat = dir([inputdir '*' '.mat']);
    N = size(InputMat,1);    % Total Training Images   
    fprintf('\nTotal Mat loaded : %d\n',N);
    
    % Create PPM Dir
    if ~exist(ppmdir,'dir')
        mkdir(ppmdir);
    end
    % Create Superpixel Dir
    if ~exist(superpixeldir,'dir')
        mkdir(superpixeldir);
    end
    % Create Result Dir
    if ~exist(resultdir,'dir')
        mkdir(resultdir);
    end
    
    TimeAnalysis = [resultdir '/TimeAnalysis'];  
    fp=fopen(TimeAnalysis,'a+');
    fprintf(fp,'ID,Name,HEIGHT,WIDTH,PPM_TIME,SUPERPIXEL_TIME,PREDICTIONTIME\n');
    fclose(fp);
    
    ResultAnalysis = [resultdir '/ResultAnalysis'];
    fp=fopen(ResultAnalysis,'a+');
    fprintf(fp,'ID,Name,Varsha_Spatial_Accuracy,Varsha_Spatial_Recall,Varsha_Spatial_Precision,Varsha_Spatial_Gmean,Varsha_Spatial_Specificity,Varsha_Spatial_tp,Varsha_Spatial_fp,Varsha_Spatial_fn,Varsha_Spatial_tn,Varsha_Spatial_total\n');
    fclose(fp);
    
    
    for ID = 1:N
        fprintf('\n---%d %s----',ID,InputMat(ID).name);
        load([inputdir InputMat(ID).name]);
        IMGNAME = [inputdir InputMat(ID).name(1:end-4) '.jpg'];
        imwrite(IMG,IMGNAME);
        PPMNAME = [ppmdir InputMat(ID).name(1:end-4) '.ppm'];
        SUPERPIXELNAME = [superpixeldir InputMat(ID).name(1:end-4) '.pnm'];
        convert='convert';
        segcmd = '/home/luminous/Desktop/Indoor_Scene_Understanding_August_2013/My_Code/ICPR_Final/Code/multipleSegmentations_superPixel/segment 0.8 100 100';
        PPM_TIME = tic;
        syscall = [convert ' ' IMGNAME ' ' PPMNAME];
        system(syscall);
        PPM_TIME = toc(PPM_TIME);
        
        SUPERPIXEL_TIME = tic;
        syscall = [segcmd ' ' PPMNAME ' ' SUPERPIXELNAME];
        system(syscall);
        SUPERPIXEL_TIME = toc(SUPERPIXEL_TIME);
        
        Name = [InputMat(ID).name(1:end-4) '.jpg'];
        
        PREDICTIONTIME = tic;
        VarshaSpatialLayout(inputdir,Name,resultdir,superpixeldir);
        PREDICTIONTIME = toc(PREDICTIONTIME);
        
        GTLABEL = double(LABEL);
        GTLABEL(GTLABEL==0) = -1;
        Varsha_Spatial = Evaluate(GTLABEL,MASK);
        SavePredictedMask(IMG,MASK,[layoutdir 'Spatial_' Name]);
        
        fp=fopen(ResultAnalysis,'a+');
        fprintf(fp,'%d,%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n',...
            ID,Name,Varsha_Spatial.Accuracy,Varsha_Spatial.Recall,Varsha_Spatial.Precision,Varsha_Spatial.Gmean,Varsha_Spatial.Specificity,Varsha_Spatial.tp,Varsha_Spatial.fp,Varsha_Spatial.fn,Varsha_Spatial.tn,Varsha_Spatial.total);
        fclose(fp);
            
        TimeAnalysis = [resultdir '/TimeAnalysis'];  
        fp=fopen(TimeAnalysis,'a+');
        fprintf(fp,'%d,%s,%d,%d,%f,%f,%f\n',ID,InputMat(ID).name,HEIGHT,WIDTH,PPM_TIME,SUPERPIXEL_TIME,PREDICTIONTIME);
        fclose(fp);
    end
    toc
end


function VarshaSpatialLayout(inputdir,Name,resultdir,superpixeldir) 
    global boxlayout surface_labels MASK layoutdir;
    [ boxlayout,surface_labels,MASK] = getspatiallayout(inputdir,Name,resultdir,superpixeldir);
    layoutdir = [resultdir '/Layouts_Labels/'];
    if ~exist(layoutdir,'dir')
        mkdir(layoutdir);
    end
    MASK(MASK==0)=-1;
    save([layoutdir Name(1:end-3) 'mat'],'boxlayout','surface_labels','MASK');
    close all;
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

function SavePredictedMask(IMG,MASK,name)    
    BLENDIMG = imoverlay(IMG,MASK==1,[0 1 0]);
    BLENDIMG = imfuse(BLENDIMG,IMG,'blend','Scaling','joint');
    imwrite(BLENDIMG,name);
end
