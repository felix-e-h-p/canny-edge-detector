%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image Corner Segmentation and Evaluation
%
% Specify image number 'imNum' and directory 'ImDir' below:
%   imNum is currently set to the default value of 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

% Image parameters
imNum = 1;
ImDir = 'Images/';

% Read the image
imFile = [ImDir, 'im', int2str(imNum), '.jpg'];
I = im2double(imread(imFile));

% Call the main segmentation function
seg = segment_image(I);

% Normalize 'seg' for binary display
seg = seg / max(seg(:));

% Load human observation segmentations
humanFiles = [ImDir, 'im', int2str(imNum), 'seg*.png'];
numFiles = length(dir(humanFiles));

for i = 1 : numFiles
    humanFile = [ImDir, 'im', int2str(imNum), 'seg', int2str(i), '.png'];
    boundariesHuman(:, :, i) = im2double(imread(humanFile));
end

% Convert seg to boundaries
boundariesPred = convert_seg_to_boundaries(seg);

% Evaluate performance
[f1score, TP, FP, FN] = evaluate(boundariesPred, boundariesHuman);

% Visualize results
show_results(boundariesPred, boundariesHuman, f1score, TP, FP, FN);
fprintf('F1 score = %f\n', round(f1score, 2));

% Canny Edge Detection Implementation
function [seg] = segment_image(I)

    % Convert image to grayscale double precision format
    Idg = rgb2gray(I);

    % Median Filter for pre-processing
    Idg = medfilt2(Idg);

    % Gaussian Filter for noise suppression
    gfc = [2,4,5,4,2;4,9,12,9,4;5,12,15,12,5;4,9,12,9,4;2,4,5,4,2];
    F = sum(gfc, 'all');
    gfc = (1/F) * gfc;
    I_g = conv2(Idg, gfc, 'same');

    % Image gradients using Sobel Filter
    dx = [-1,0,1;-2,0,2;-1,0,1];
    dy = [1,2,1;0,0,0;-1,-2,-1];
    I_dx = conv2(I_g, dx, 'same');
    I_dy = conv2(I_g, dy, 'same');
    I_mag = sqrt((I_dx.^2) + (I_dy.^2));

    % Non-maximum suppression
    % ...

    % Double level thresholding
    % ...

    % Hysteresis thresholding
    % ...

    % Canny edge detector implementation
    c_edges = T_res * 255;
    seg = c_edges;

end

% Convert region labels to boundaries
function b = convert_seg_to_boundaries(seg)

    seg = padarray(seg, [1, 1], 'post', 'replicate');
    b = abs(conv2(seg, [-1,1], 'same')) + abs(conv2(seg, [-1;1], 'same')) + ...
        abs(conv2(seg, [-1,0;0,1], 'same')) + abs(conv2(seg, [0,-1;1,0], 'same'));
    b = im2bw(b(1:end-1,1:end-1), 0);

end

% Evaluate Canny Detector with Median Smoothing
function [f1score, TP, FP, FN] = evaluate(boundariesPred, boundariesHuman)

    % Set tolerance for boundary matching
    r = 16;
    neighbourhood = strel('disk', r, 0);

    % Thin boundaries
    boundariesPredThin = boundariesPred .* bwmorph(boundariesPred, 'thin', inf);
    boundariesHumanThin = prod(imdilate(boundariesHuman, neighbourhood), 3);
    boundariesHumanThin = boundariesHumanThin .* bwmorph(boundariesHumanThin, 'thin', inf);

    % Thick boundaries
    boundariesPredThick = imdilate(boundariesPred, neighbourhood);
    boundariesHumanThick = max(imdilate(boundariesHuman, neighbourhood), [], 3);

    % True Positives, False Positives, False Negatives
    TP = boundariesPredThin .* boundariesHumanThick;
    FP = max(0, boundariesPred - boundariesHumanThick);
    FN = max(0, boundariesHumanThin - boundariesPredThick);

    % Calculate F1 score
    numTP = sum(TP(:));
    numFP = sum(FP(:));
    numFN = sum(FN(:));
    f1score = 2 * numTP / (2 * numTP + numFP + numFN);

end

% Show comparison between predicted and human image segmentations
function show_results(boundariesPred, boundariesHuman, f1score, TP, FP, FN)

    figure();

    % Predicted Boundaries
    maxsubplot(2, 2, 3);
    imagescc(boundariesPred);
    title('Predicted Boundaries');
    [a, b] = size(boundariesPred);
    if a > b
        ylabel(['F1 Score=', num2str(f1score, 2)]);
    else
        xlabel(['F1 Score=', num2str(f1score, 2)]);
    end

    % Human Boundaries
    maxsubplot(2, 2, 4);
    imagescc(mean(boundariesHuman, 3));
    title('Human Boundaries');

    % True Positives, False Positives, False Negatives
    maxsubplot(2, 3, 1);
    imagescc(TP);
    title('True Positives');
    maxsubplot(2, 3, 2);
    imagescc(FP);
    title('False Positives');
    maxsubplot(2, 3, 3);
    imagescc(FN);
    title('False Negatives');

    colormap('gray');
    drawnow;

end

% Combine imagesc with other commands for improved appearance
function imagescc(I)

    imagesc(I, [0, 1]);
    axis('equal', 'tight');
    set(gca, 'XTick', [], 'YTick', []);

end

% Set position for subplot
function position = maxsubplot(rows, cols, ind, fac)

    if nargin < 4
        fac = 0.075;
    end

    position = [(fac/2)/cols + rem(min(ind)-1, cols)/cols, ...
                (fac/2)/rows + fix((min(ind)-1)/cols)/rows, ...
                (length(ind)-fac)/cols, (1-fac)/rows];
    axes('Position', position);

end
