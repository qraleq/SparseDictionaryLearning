clearvars
close all
clc

image = im2double(imread('lena.png'));
% image = imgaussfilt(image, 1.2);

% image = rgb2gray(image);

% image=imresize(imbinarize(checkerboard(64)),8);

bb = 7;

imageBlocks = im2col(image, [bb, bb], 'sliding');

averagedImageBlocks = sum(imageBlocks, 1)./(bb.^2);

imageBlocksSubtracted = imageBlocks - repmat(averagedImageBlocks, (bb.^2), 1);

[V, D, W] = eig(cov(imageBlocksSubtracted'));

for i=1:size(V,2)
    mask(:,:,i) = reshape(W(:,i), [bb, bb]);
    
    filteredImages(:,:,i) = conv2(image, mask(:,:,i), 'same');
    
    figure(1)
    imshow(filteredImages(:,:,i))
    drawnow
%         waitforbuttonpress
    
    
end

edgeMap = max((filteredImages(:,:, 1:end)), [], 3);
% edgeMap = mean2((filteredImages(:,:, 1:end)), 3);

% se = strel('disk', 2);
% edgeMap = imerode(edgeMap, se);

[Ix, Iy] = gradient(filteredImages(:,:,i));

I = hypot(Ix, Iy);

% edgeMap=I;

orient = atan2(Ix, Iy);
orient(orient < 0) = orient(orient < 0) + pi;

% orient = smoothorient(orient, 2);

orient = int8(rad2deg(orient));

radius = 1.5;

edgeMap = nonmaxsup(edgeMap, orient, radius);

threshold = 5*mean2(edgeMap);

edgeMap = hysthresh(edgeMap, threshold, 0.95*threshold);

% edgeMap = imbinarize(edgeMap, mean(edgeMap(:)));

% edgeMap= bwmorph(edgeMap, 'close', 100);

edgeMapMatlab = edge(image, 'Canny');

figure, colormap gray
subplot(121), imshow(edgeMap)
subplot(122), imshow(edgeMapMatlab)
title('Detected Edges')


