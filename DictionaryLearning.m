%% DICTIONARY LEARNING
clearvars
close all
clc

%% INITIALIZATION
addpath('utilities')
addpath('data')

% size of extracted square (w*w) patches
blockSize = 16;

% number of image patches in set Y
N = 1000;

% length of signal y (vectorized image patch)
n = blockSize^2;

% desired sparsity (number of non-zero elements in sparse representation vector)
T0 = 20;

% number of atoms in dictionary D
K = 324;

% load image for patch extraction
imagePath = '.\data\barb.png';
image = im2double(imresize(imread(imagePath), 1));

% add additive noise noise
sigma = 0.1;
image = image + sigma*randn(size(image));

[imH, imW] = size(image);

%% EXTRACT IMAGE PATCHES & INITIALIZE DICTIONARY D0 & PLOT DICTIONARY

[~, Y] = extractImagePatches(image, blockSize, 'rand', 'nBlocks', 1000);
Y = Y - repmat(mean(Y, 1), [blockSize^2,1]);

D0 = initDictionaryFromPatches(n, K, Y);

% ALTERNATIVE: generate overcomplete DCT dictionary
% D0 = overcompleteDCTdictionary(n, p);

visualizeDictionary(D0);
title('Initial Dictionary')

%% CALCULATE COEFFICIENTS X
D = D0;
X = zeros(size(D, 2), size(Y, 2));

X = sparseCode(Y, X, D, T0, 20, 'Plot', 0);

%% UPDATE DICTIONARY D
D = D0;

D = updateDictionary(Y, X, D, 'ksvd', 'nIter', 15, 'Plot', 0);


%% DICTIONARY LEARNING
% perform dictionary learning by iteratively repeating coefficient
% calculation and dictionary update steps

niter_learn = 20;
niter_coeff = 10;
niter_dict = 10;

D = D0;
X = zeros(size(D, 2), size(Y, 2));
E0 = [];

sigma = 0.1;
lambda = 1.5 * sigma;


for iter = 1:niter_learn
    fprintf('Dictionary Learning Iteration No. %d\n', iter);

    %%%%%%%%%%%%%%%% coefficient calculation %%%%%%%%%%%%%%%%%%%%%%%
    X = sparseCode(Y, X, D, T0, niter_coeff);
    
    E0(end+1) = norm(Y-D*X, 'fro')^2;
    
    %%%%%%%%%%%%%%%% dictionary update %%%%%%%%%%%%%%%%%%%%%%%%%%%
    [D, X] = updateDictionary(Y, X, D, 'ksvd', 'nIter', niter_dict);

    E0(end+1) = nor(Y-D*X, 'fro')^2;
end


figure,
hold on
plot(1:2*niter_learn, E0);
plot(1:2:2*niter_learn, E0(1:2:2*niter_learn), '*');
plot(2:2:2*niter_learn, E0(2:2:2*niter_learn), 'o');
axis tight;
legend('|Y-DX|^2', 'After coefficient update', 'After dictionary update');

%% DICTIONARY LEARNING - SPAMS

param.lambda = 0.1;
param.numThreads = -1; % number of threads
param.iter = 50;  % let us see what happens after 1000 iterations.
param.mode = 5;
param.D = D0;

D = mexTrainDL(Y, param);

%%

visualizeDictionary(D)
title('Final Dictionary')

%%

[x_rec, r, coeff, iopt] = wmpalg('OMP', Y(:,1), D0, 'itermax', 100, 'maxerr', {'L2', 20});

iopt



