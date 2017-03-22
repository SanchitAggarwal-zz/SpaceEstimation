function changeImg(inputdir)
    %addpaths;
    InputMat = dir([inputdir '*' '.jpg']);
    N = size(InputMat,1);    % Total Training Images   
    fprintf('\nTotal Mat loaded : %d\n',N);
       
    for ID = 1:N
        IMGNAME = [inputdir InputMat(ID).name];
        IMG = imread([inputdir InputMat(ID).name]);
%         convert='convert -resize 25%';
%         syscall = [convert ' ' IMGNAME ' ' IMGNAME];
%         system(syscall);
        IMG = imresize(IMG,[512 512]);
         imwrite(IMG,IMGNAME);
    end
end