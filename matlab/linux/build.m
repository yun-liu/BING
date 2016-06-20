clear; clc;

% change these paths to yours
path_to_BING = '../../linux/BING/';
path_to_OpenCV = '/usr/local/';
path_to_openmp = '/usr/lib/gcc/x86_64-linux-gnu/4.9/';

% copy windows code into current folder
copyfile(fullfile(path_to_BING,'*.h'),'.','f');
copyfile(fullfile(path_to_BING,'*.cpp'),'.','f');

% run build command
mexCommand = 'mex mexBING.cpp FilterBING.cpp ';
mexCommand = [mexCommand '-I' path_to_OpenCV 'include '];
mexCommand = [mexCommand '-L' path_to_OpenCV 'lib -L' path_to_openmp ' '];
mexCommand = [mexCommand '-lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_ts -lopencv_imgcodecs -lgomp'];
eval(mexCommand);

% remove windows code copied just now
inc_files = dir(fullfile(path_to_BING,'*.h'));
src_files = dir(fullfile(path_to_BING,'*.cpp'));
for i = 1:length(inc_files)
    delete(inc_files(i).name);
end
for i = 1:length(src_files)
    delete(src_files(i).name);
end