%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Specify image number 'imNum' and directory 'ImDir' below - lines 8-9
% imNum is currently at default value of 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

imNum = 1;
ImDir = 'Images/';

imFile = [ImDir, 'im', int2str(imNum), '.jpg'];

I = im2double(imread(imFile));

% Call main function to determine edge locations:

seg = segment_image(I);

% Normalise 'seg' to display binary values:

seg = seg/max(seg(:));

% Load human observation segmentations:

humanFiles = [ImDir, 'im', int2str(imNum), 'seg*.png'];
numFiles = length(dir(humanFiles));

for i = 1 : numFiles
    humanFile = [ImDir, 'im', int2str(imNum), 'seg', int2str(i), '.png'];
    boundariesHuman(:,:,i) = im2double(imread(humanFile));
end

% Convert seg to boundaries:

boundariesPred = convert_seg_to_boundaries(seg);

% Performance evaluation stage:

[f1score, TP, FP, FN] = evaluate(boundariesPred, boundariesHuman);

% Visulisation of results, both graphically and numerically:

show_results(boundariesPred, boundariesHuman, f1score, TP, FP, FN);
fprintf('F1 score = %f\n', round(f1score, 2));

% Implementation of Canny Edge Detection, through a multistage methodology,
% in order to sufficiently detect and henceforth segment relevant and 
% required edges from multiple and varied RGB colourspace images.

function [seg] = segment_image(I);

% Convert image to grayscale double precision format:

Idg = rgb2gray(I);

% Pre-processing stage via implemenation of a Median Filter:

Idg = medfilt2(Idg);

% Pre-processing stage via implemenation of a Gaussian Filter for noise
% suppression - coefficients formally specified as follows:

gfc = [2,4,5,4,2;4,9,12,9,4;5,12,15,12,5;4,9,12,9,4;2,4,5,4,2];
F = sum(gfc, 'all');
gfc = (1/F).*gfc;

I_g = conv2(Idg, gfc, 'same');

I_size = size(I_g);
n_rows = I_size(1);
n_cols = I_size(2);

% Calculation of image gradients - i.e. horizontal and vertical components 
% of the original image, using the basis of Sobel Filter methodology in 
% order to obtain the edge gradient value for each pixel of 'I_g':

dx = [-1,0,1;-2,0,2;-1,0,1];
dy = [1,2,1;0,0,0;-1,-2,-1];

I_dx = conv2(I_g, dx, 'same');
I_dy = conv2(I_g, dy, 'same');

I_mag = sqrt((I_dx.^2) + (I_dy.^2));

% Non maximum supression stage, preceeded by a very basic image gradient 
% quantisation stage - adjusting for negative pixel values:

I_angle = atan2(I_dy, I_dx);

M = zeros(n_rows, n_cols);

for i = 2 : n_rows - 1
    for j = 2 : n_cols - 1
        if (I_angle(i,j) == 0)
            M(i, j) = (I_mag(i, j) == max([I_mag(i, j), I_mag(i, j+1), I_mag(i, j-1)]));
        elseif I_angle(i,j) == 45
            M(i, j) = (I_mag(i, j) == max([I_mag(i, j), I_mag(i+1, j-1), I_mag(i-1, j+1)]));
        elseif I_angle(i,j) == 90
            M(i, j) = (I_mag(i, j) == max([I_mag(i, j), I_mag(i+1, j), I_mag(i-1, j)]));
        elseif I_angle(i,j) == 135
            M(i, j) = (I_mag(i, j) == max([I_mag(i, j), I_mag(i+1, j+1), I_mag(i-1, j-1)]));
        end
    end
end

% Double level thresholding input specification - must be noted that an
% optimal vale of T_l can be specified based on minimisation of interclass
% variance:

[counts, x] = imhist(Idg, 64);
T_o = otsuthresh(counts);

T_l = (1/10.)*(graythresh(Idg));
T_h = T_o.*0.01;

% Hysteresis thresholding for final edge tracking - magnitude of pixel
% gradients directly compared with the pre-defined input threshold values:

T_l = T_l * max(max(I_angle));
T_h = T_h * max(max(I_angle));
T_res = zeros(n_rows, n_cols);

for i = 1 : n_rows
    for j = 1 : n_cols
        if (I_mag(i, j) < T_l)
            T_res(i, j) = 0;
        elseif (I_mag(i, j) > T_h)
            T_res(i, j) = 1;
        elseif (I_mag(i+1, j) > T_h) 
            T_res(i, j) = 1;
        elseif (I_mag(i-1, j) > T_h) 
            T_res(i, j) = 1;
        elseif (I_mag(i, j+1) > T_h) 
            T_res(i, j) = 1;
        elseif (I_mag(i, j-1) > T_h) 
            T_res(i, j) = 1;
        elseif (I_mag(i+1, j+1) > T_h) 
            T_res(i, j) = 1;
        elseif (I_mag(i-1, j+1) > T_h) 
            T_res(i, j) = 1;
        elseif (I_mag(i+1, j-1) > T_h) 
            T_res(i, j) = 1;
        elseif (I_mag(i-1, j-1) > T_h) 
            T_res(i, j) = 1;
        end
    end
end

% Canny edge detector implementation and optional visulisation stage:

c_edges = T_res.*255;
seg = c_edges;

% figure(), clf; imshow(uint8(c_edges));

end

% Performs conversion from an array containing defined region labels 'seg' 
% to one containing the boundaries between the regions 'b':

function b = convert_seg_to_boundaries(seg)

seg = padarray(seg,[1,1],'post','replicate');
b = abs(conv2(seg,[-1,1],'same'))+abs(conv2(seg,[-1;1],'same'))+abs(conv2(seg,[-1,0;0,1],'same'))+abs(conv2(seg,[0,-1;1,0],'same'));
b = im2bw(b(1:end-1,1:end-1),0);

end

% Evaluation of Canny Detector with Median Smoothing when directly compared
% to predefined human detected image boundaries:

function [f1score,TP,FP,FN] = evaluate(boundariesPred,boundariesHuman)

% Set tolerance for boundary matching:

r = 16;
neighbourhood = strel('disk',r,0); 

boundariesPredThin = boundariesPred.*bwmorph(boundariesPred,'thin',inf);
boundariesHumanThin = prod(imdilate(boundariesHuman,neighbourhood),3);
boundariesHumanThin = boundariesHumanThin.*bwmorph(boundariesHumanThin,'thin',inf);
boundariesPredThick = imdilate(boundariesPred,neighbourhood);
boundariesHumanThick = max(imdilate(boundariesHuman,neighbourhood),[],3);

TP=boundariesPredThin.*boundariesHumanThick;
FP=max(0,boundariesPred-boundariesHumanThick);
FN=max(0,boundariesHumanThin-boundariesPredThick);

numTP=sum(TP(:));
numFP=sum(FP(:));
numFN=sum(FN(:));

f1score=2*numTP/(2*numTP+numFP+numFN);

end

% Function used to show comparison between predicted and human image
% segmentations:

function show_results(boundariesPred,boundariesHuman,f1score,TP,FP,FN)

figure()

maxsubplot(2,2,3); imagescc(boundariesPred); title('Predicted Boundaries')
[a,b]=size(boundariesPred);

if a>b
    ylabel(['f1score=',num2str(f1score,2)]);
else
    xlabel(['f1score=',num2str(f1score,2)]);
end

maxsubplot(2,2,4); imagescc(mean(boundariesHuman,3)); title('Human Boundaries')
maxsubplot(2,3,1); imagescc(TP); title('True Positives')
maxsubplot(2,3,2); imagescc(FP); title('False Positives')
maxsubplot(2,3,3); imagescc(FN); title('False Negatives')
colormap('gray'); 
drawnow;

end

% Combines imagesc with some other commands to improve appearance of 
% images:

function imagescc(I)

imagesc(I,[0,1]); 
axis('equal','tight'); 
set(gca,'XTick',[],'YTick',[]);

end

function position = maxsubplot(rows,cols,ind,fac)

if nargin<4, fac=0.075; end

position=[(fac/2)/cols+rem(min(ind)-1,cols)/cols,...
          (fac/2)/rows+fix((min(ind)-1)/cols)/rows,...
          (length(ind)-fac)/cols,(1-fac)/rows];
axes('Position',position);

end
