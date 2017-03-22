function GenerateHTML(InputDir,OutputDir,FileName)
    addpaths;
    if ~exist(InputDir,'dir')
       error('\n%s','Input not exist');
    else
        Images = dir([InputDir '*' '_BOUNDARY.jpg']);
    end
    % Create Output Dir
    if ~exist(OutputDir,'dir')
        mkdir(OutputDir);
    end
    %POSTFIX =['.jpg         ';
    POSTFIX = ['_GT.jpg      ';
               '_SVM.jpg     ';
               '_POSITION.jpg';
               '_BOUNDARY.jpg';
               '_GENERIC.jpg ';
               '_GRABCUT.jpg '];
               '_GRABCUT_PIXEL.jpg   '];
    POSTFIX = cellstr(POSTFIX);
    N = size(Images,1);    % Total Tested Images Mat files
    fprintf('\nTotal MAT loaded : %d\n',N);
    HTML = [OutputDir '/' FileName];
    fp=fopen(HTML,'a+');    
    StartingHTML(fp);    
    for ID = 1:N
        NAME=strrep(Images(ID).name,'_BOUNDARY.jpg','');        
        fprintf(fp,'<hr>\n');
        fprintf(fp,'<div class="row">\n');
        for i = 1:size(POSTFIX,1)
            ImageHTML(fp,[InputDir NAME cell2mat(POSTFIX(i,:))]);
        end
        fprintf(fp,'</div>\n');
    end
    EndingHTML(fp);
    fclose(fp);
end
function StartingHTML(fp)

    fprintf(fp,'<!DOCTYPE html>\n');
    fprintf(fp,'<html lang="en">\n');
    fprintf(fp,'<head>\n');
    fprintf(fp,'<meta charset="utf-8">\n');
    fprintf(fp,'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n');
    fprintf(fp,'<meta name="description" content="">\n');
    fprintf(fp,'<meta name="author" content="">\n');

    fprintf(fp,'<title>Floor Segmentation</title>\n');

    fprintf(fp,'<!-- Bootstrap core CSS -->\n');
    fprintf(fp,'<link href="css/bootstrap.css" rel="stylesheet">\n');

    fprintf(fp,'<link href="css/thumbnail-gallery.css" rel="stylesheet">\n');
    fprintf(fp,'</head>\n');

    fprintf(fp,'<body>\n');

    fprintf(fp,'<nav class="navbar navbar-fixed-top navbar-inverse" role="navigation">\n');
    fprintf(fp,'<div class="container">\n');
    fprintf(fp,'<div class="navbar-header">\n');
    fprintf(fp,'<button type="button" class="navbar-toggle" data-toggle="collapse" data-target=".navbar-ex1-collapse">\n');
    fprintf(fp,'<span class="sr-only">Toggle navigation</span>\n');
    fprintf(fp,'<span class="icon-bar"></span>\n');
    fprintf(fp,'<span class="icon-bar"></span>\n');
    fprintf(fp,'<span class="icon-bar"></span>\n');
    fprintf(fp,'</button>\n');
    fprintf(fp,'<a class="navbar-brand">Floor Segmentation</a>\n');
    fprintf(fp,'</div>\n');

    fprintf(fp,'<!-- Collect the nav links, forms, and other content for toggling -->\n');
    fprintf(fp,'<div class="collapse navbar-collapse navbar-ex1-collapse">\n');
    fprintf(fp,'<ul class="nav navbar-nav">\n');
    fprintf(fp,'<li><a href="Experiment1.html">Experiment 1</a></li>\n');
    fprintf(fp,'<li><a href="Experiment2.html">Experiment 2</a></li>\n');
    fprintf(fp,'<li><a href="Experiment3.html">Experiment 3</a></li>\n');
    fprintf(fp,'<li><a href="Experiment4.html">Experiment 4</a></li>\n');
    fprintf(fp,'<li><a href="Experiment5.html">Experiment 5</a></li>\n');
    fprintf(fp,'<li class="active"><a href="Experiment6_Qualitative.html">Experiment 6</a></li>\n');
    fprintf(fp,'<li><a href = "http://researchweb.iiit.ac.in/~sanchit.aggarwal/ViShruti/ViShruti.html">Navigation With Sound</a></li>\n');
    fprintf(fp,'</ul>\n');
    fprintf(fp,'</div><!-- /.navbar-collapse -->\n');
    fprintf(fp,'</div><!-- /.container -->\n');
    fprintf(fp,'</nav>\n');

    fprintf(fp,'<div class="container">\n');

    fprintf(fp,'<div class="row">\n');
    fprintf(fp,'<div class="col-lg-6">\n');
    fprintf(fp,'<h1 class="page-header">Qualitative Results Images 512x512 with 400 pixels in each superpixel </h1>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'<div class="row">\n');
    fprintf(fp,'<div class="col-lg-12">\n');
    fprintf(fp,'<div class="col-lg-2 col-md-4 col-xs-6">\n');
    fprintf(fp,'<h5 style="text-align: justify">Groundtruth</h5>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'<div class="col-lg-2 col-md-4 col-xs-6">\n');
    fprintf(fp,'<h5 style="text-align: justify">SVM Floor Estimate</h5>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'<div class="col-lg-2 col-md-4 col-xs-6">\n');
    fprintf(fp,'<h5 style="text-align: justify">KDE Floor Estimate</h5>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'<div class="col-lg-2 col-md-4 col-xs-6">\n');
    fprintf(fp,'<h5 style="text-align: justify">Boundary Floor Estimate</h5>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'<div class="col-lg-2 col-md-4 col-xs-6">\n');
    fprintf(fp,'<h5 style="text-align: justify">SVM+KDE+Boundary Floor Estimate</h5>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'<div class="col-lg-2 col-md-4 col-xs-6">\n');
    fprintf(fp,'<h5 style="text-align: justify">GrabCut Estimate(Final Segmentation)</h5>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'</div>\n');
    fprintf(fp,'</div>\n');
      
end
function ImageHTML(fp,Name)
    fprintf(fp,'<div class="col-lg-1 col-md-4 col-xs-6 thumb">\n');
    fprintf(fp,'<a class="thumbnail" href="%s">\n',Name);
    fprintf(fp,'<img class="img-responsive" src="%s">\n',Name);
    fprintf(fp,'</a>\n');
    fprintf(fp,'</div>\n');
end
function EndingHTML(fp)
    fprintf(fp,'</div><!-- /.container -->\n');
    fprintf(fp,'<!-- JavaScript -->\n');
    fprintf(fp,'<script src="js/jquery-1.10.2.js"></script>\n');
    fprintf(fp,'<script src="js/bootstrap.js"></script>\n');
    fprintf(fp,'</body>\n');
    fprintf(fp,'</html>\n');
end
