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
% sigma = 0.1;
% image = image + sigma*randn(size(image));

[imH, imW] = size(image);

%% EXTRACT IMAGE PATCHES & INITIALIZE DICTIONARY D0 & PLOT DICTIONARY

[~, Y] = extractImagePatches(image, blockSize, 'rand', 'nPatches', 1000);

% Y = kron(dctmtx(16),dctmtx(16))*Y;

Y = Y - repmat(mean(Y, 1), [blockSize^2,1]);

D0 = initDictionaryFromPatches(n, K, Y);

% ALTERNATIVE: generate overcomplete DCT dictionary
% D0 = overcompleteDCTdictionary(n, K);

visualizeDictionary(D0);
title('Initial Dictionary')

%% CALCULATE COEFFICIENTS X
D = D0;
X = zeros(size(D, 2), size(Y, 2));

X = sparseCode(Y, D, T0, 20, 'Plot', 0);

%% UPDATE DICTIONARY D
D = D0;

D = updateDictionary(Y, X, D, 'ksvd', 'nIter', 15, 'Plot', 0, 'Verbose', 1);


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


[D, X, E0] = learnDictionary(Y, D, T0, 'Plot', 1);


%% DICTIONARY LEARNING - SPAMS

param.lambda = 0.1;
param.numThreads = -1; % number of threads
param.iter = 50;  % let us see what happens after 1000 iterations.
param.mode = 5;
param.D = D0;

D = mexTrainDL(Y, param);

%%
figure
visualizeDictionary(D)
title('Final Dictionary')

%%

[x_rec, r, coeff, iopt] = wmpalg('OMP', Y(:,1), D0, 'itermax', 100, 'maxerr', {'L2', 20});

iopt



