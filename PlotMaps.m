function PlotMaps(InputDir,OutputDir)
    addpaths;
    time = tic;
    fnName = 'ResultAnalysis:';
    msg = '----------begins----------';
    fprintf('\n%s %s',fnName,msg);
    if ~exist(InputDir,'dir')
       error('\n%s',msg);
    else
        INPUTDIR = InputDir;
        OutputImages = dir([OutputDir '*' '_BOUNDARY.jpg']);
    end
    % Create Output Dir
    if ~exist(OutputDir,'dir')
        mkdir(OutputDir);
    end
    
    N = size(OutputImages,1);    % Total Tested Images Mat files
    fprintf('\nTotal MAT loaded : %d\n',N);
    for ID = 1:N
        fprintf('\n---%d %s----',ID,OutputImages(ID).name(1:end-13));
        s = load([InputDir OutputImages(ID).name(1:end-13) '.mat']);
               
        %Plot Boundaries
        f=figure;
        set(f, 'visible', 'off');
        imagesc(s.IMG);
        colormap(gray);
        hold on;
        plot(s.lines(s.VL, [1 2])', s.lines(s.VL, [3 4])','b');
        plot(s.lines(s.HL, [1 2])', s.lines(s.HL, [3 4])','g');
        plot(s.lines(s.IL, [1 2])', s.lines(s.IL, [3 4])','r');
        plot(s.PolyX,s.PolyY,'y');
        plot(s.FPolyX,s.FPolyY,'c');
        saveas(f,[OutputDir OutputImages(ID).name(1:end-13) '_LINES.jpg']);        
        
    end
    toc(time);
end
