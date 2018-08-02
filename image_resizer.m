addpath 'D:\RIT\631\Project\data_project\temp\Violence\';
imagefiles = dir('D:\RIT\631\Project\data_project\temp\Violence\*.jpg');      
nfiles = length(imagefiles);    % Number of files found
for index=1:1
   currentfilename = imagefiles(index).name;
   disp(size(imagefiles(index)));
   currentimage = imread(currentfilename);
   J = imresize(currentimage, 0.2);
   s=strcat(int2str(index),currentfilename);
   imwrite(J, s);
end
