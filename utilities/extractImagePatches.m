function [ patches ] = extractImagePatches( image, blockSize )
%extractImagePatches Summary of this function goes here
%   Detailed explanation goes here

[imH, imW] = size(image);

% overlap
q = blockSize;

[y, x] = meshgrid(1:q:imW-blockSize/2, 1:q:imH-blockSize/2);
[dY,dX] = meshgrid(0:blockSize-1,0:blockSize-1);

m = size(x(:),1);

% create indexing grids for block extraction
Xp = repmat(dX,[1 1 m]) + repmat( reshape(x(:),[1 1 m]), [blockSize blockSize 1]);
Yp = repmat(dY,[1 1 m]) + repmat( reshape(y(:),[1 1 m]), [blockSize blockSize 1]);

% boundary indices condition 
Xp(Xp>imH) = 2*imH-Xp(Xp>imH);
Yp(Yp>imW) = 2*imW-Yp(Yp>imW);

Y = image(Xp+(Yp-1)*imH);
Y = reshape(Y, [blockSize^2, m]);





% number of randomly selected image patches
q = 3*N;

% select q random locations in image (upper left block corners)
x = floor(rand(1,1,q)*(imH-blockSize))+1;
y = floor(rand(1,1,q)*(imW-blockSize))+1;

% create rectangular mesh wxw
[dY,dX] = meshgrid(0:blockSize-1,0:blockSize-1);

% generate matrices containting block locations
Xp = repmat(dX, [1 1 q]) + repmat(x, [blockSize blockSize 1]);
Yp = repmat(dY, [1 1 q]) + repmat(y, [blockSize blockSize 1]);

% extract and vectorize blocks
Y = image(Xp+(Yp-1)*imW);
Y = reshape(Y, [n, q]);


end

