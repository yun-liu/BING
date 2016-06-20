clear; clc;

% change these paths to yours
path_to_BING = '../../windows/BING/';
path_to_OpenCV = 'C:/SDK/OpenCV/opencv3_0_0/build/';

% copy windows code into current folder
copyfile(fullfile(path_to_BING,'*.h'),'.','f');
copyfile(fullfile(path_to_BING,'*.cpp'),'.','f');

% run build command
mexCommand = 'mex mexBING.cpp FilterBING.cpp ';
mexCommand = [mexCommand '-I' path_to_OpenCV 'include '];
mexCommand = [mexCommand '-L' path_to_OpenCV 'x64/vc12/lib '];
mexCommand = [mexCommand '-lopencv_ts300 -lopencv_world300'];
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