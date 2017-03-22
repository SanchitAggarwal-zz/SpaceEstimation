function GenerateLatex(InputDir,ImageDir,OutputDir,FileName)
    addpaths;
    if ~exist(InputDir,'dir')
       error('\n%s',msg);
    else
        Images = dir([InputDir '*' '_BOUNDARY.jpg']);
    end
    % Create Output Dir
    if ~exist(OutputDir,'dir')
        mkdir(OutputDir);
    end
    POSTFIX = ['.jpg              ';
               '_GT.jpg           ';
               '_SVM.jpg          ';
               '_POSITION.jpg     ';
               '_BOUNDARY.jpg     ';
               '_GENERIC.jpg      ';
               '_GRABCUT.jpg      ';
               '_GRABCUT_PIXEL.jpg';
               '_FINAL.jpg        '];
    POSTFIX = cellstr(POSTFIX);
    N = size(Images,1);    % Total Tested Images Mat files
    fprintf('\nTotal MAT loaded : %d\n',N);
    Latex = [OutputDir '/' FileName];
    fp=fopen(Latex,'a+');    
    fprintf(fp,'\\begin{figure*}\n\\begin{center}\n');
    for ID = 1:N
        if(mod(ID,10)==0)          
            fprintf(fp,'\\caption{Column [1-7]:Detected Floor after every Step, ');
            fprintf(fp,'[1]:Original Image, [2] Superpixel GroundTruth, [3]:Floor');
            fprintf(fp,' Detected by PEGASOS SVM, [4]:Floor from Kernel Density Estimate, ');
            fprintf(fp,'[5]:Floor from Boundary Detection, [6]: SVM+KDE+Boundary Floor, ');
            fprintf(fp,'[7]: GrabCut, Generic image is used as mask for grabcut}\n');
            fprintf(fp,'\\label{fig_sim}\n\\end{center}\n\\end{figure*}\n\n\n');
            fprintf(fp,'\\begin{figure*}\n\\begin{center}\n');
        end
        NAME=strrep(Images(ID).name,'_BOUNDARY.jpg','');
        
        fprintf(fp,'\\subfigure{\\includegraphics[width=.7in,height=.7in]{%s}\\label{Image}}\n',...
            [ImageDir NAME cell2mat(POSTFIX(1,:))]);
        for i = 2:size(POSTFIX,1)-1
            fprintf(fp,'\\subfigure{\\includegraphics[width=.7in,height=.7in]{%s}\\label{Image}}\n',...
            [InputDir NAME cell2mat(POSTFIX(i,:))]);
        end
        fprintf(fp,'\\subfigure{\\includegraphics[width=.7in,height=.7in]{%s}\\label{Image}}\\\\[4pt]\n\n\n',...
            [InputDir NAME cell2mat(POSTFIX(end,:))]);
        
    end
    fprintf(fp,'\\caption{Column [1-7]:Detected Floor after every Step, ');
    fprintf(fp,'[1]:Original Image, [2] Superpixel GroundTruth, [3]:Floor');
    fprintf(fp,' Detected by PEGASOS SVM, [4]:Floor from Kernel Density Estimate, ');
    fprintf(fp,'[5]:Floor from Boundary Detection, [6]: SVM+KDE+Boundary Floor, ');
    fprintf(fp,'[7]: GrabCut, Generic image is used as mask for grabcut}\n\n\n');
    fprintf(fp,'\\label{fig_sim}\n\\end{center}\n\\end{figure*}\n');
    fclose(fp);
end