clear; clc;

% the path of dataset
path = 'C:/WkDir/DetectionProposals/VOC2007/';

% get the file names in the test set
iids = {};
fid = fopen(fullfile(path,'ImageSets','Main','test.txt'),'r');
line = fgetl(fid);
while isstr(line)
    iids{end+1} = line;
    line = fgetl(fid);
end

% compute object proposals
boxes = cell(length(iids),1);
parfor i = 1:length(iids)
    I = imread(fullfile(path,'JPEGImages',[iids{i},'.jpg']));
    boxes{i} = mexBING(I,fullfile(path,'Results','ObjNessB2W8MAXBGR'));
end

% save boxes into txt files
mkdir(fullfile(path,'Results','BBoxesB2W8MAXBGR'));
parfor i = 1:length(iids)
    fid = fopen(fullfile(path,'Results','BBoxesB2W8MAXBGR',[iids{i},'.txt']),'w');
    [m,n] = size(boxes{i});
    fprintf(fid,'%d\r\n',m);
    for j = 1:m
        fprintf(fid,'%d %d %d %d\r\n',boxes{i}(j,1),boxes{i}(j,2),boxes{i}(j,3),boxes{i}(j,4));
    end
    fclose(fid);
end