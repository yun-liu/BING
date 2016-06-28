%
%  mexBINGpp is the wrapper of BING++ algorithm. Input an image, its
%  output is a N x 4 matrix, where N is the number of proposals. Each
%  row of boxes is a box with order (x1,y1,x2,y2). 
%       I   :   an RGB image
%       path:   path to trained model
%  e.g. 
%       I = imread('000001.jpg');
%       boxes{i} = mexBINGpp(I,'D:/VOC2007/Results/ObjNessB2W8MAXBGR');
%
function boxes = mexBING(I,path);