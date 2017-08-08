%% LEARNING A LINEAR CLASSIFIER WITH KSVD
close all
clearvars
clc

%% INITIALIZATION

addpath('utilities')
addpath('data')

% size of extracted square (w*w) patch
blockSize = 32;

% length of signal y (vectorized image patch)
n = blockSize^2;

Y_cat = [];
D_cat = [];


%%
% desired sparsity (number of non-zero elements in sparse representation vector)
T0 = 5;

% number of atoms in dictionary D
K = 64;

% load image for patch extraction
imagePath = '.\data\textures\1.1.12.tiff';
% imagePath = '.\data\barb.png';

image = im2double(imresize(imread(imagePath), 0.5));

% add additive noise noise
% sigma = 0.1;
% im = im + sigma*randn(size(im));

[imH, imW] = size(image);

[~, Y, Xp, Yp] = extractImagePatches(image, blockSize, 'rand', 'nPatches', 1000);

Y = Y - repmat(mean(Y), [n,1]);


D0 = initDictionaryFromPatches(n, K, Y);

[~, Y, Xp, Yp] = extractImagePatches(image, blockSize, 'seq', 'Overlap', 0);

meanY = mean(Y);
Y = Y - repmat(mean(Y), [n,1]);

% DICTIONARY LEARNING
% perform dictionary learning by iteratively repeating coefficient
% calculation and dictionary update steps
niter_learn = 3;
niter_coeff = 10;
niter_dict = 10;

D = D0;
X = zeros(size(D, 2), size(Y, 2));

for iter = 1:niter_learn
    fprintf('Dictionary Learning Iteration No. %d\n', iter);

    %%%%%%%%%%%%%%%% coefficient calculation %%%%%%%%%%%%%%%%%%%%%%%
    X = sparseCode(Y, X, D, T0, niter_coeff, 'Verbose', 0, 'StepSize', 10000);
    
   
    %%%%%%%%%%%%%%%% dictionary update %%%%%%%%%%%%%%%%%%%%%%%%%%%
    [D, X] = updateDictionary(Y, X, D, 'ksvd', 'nIter', niter_dict, 'Verbose', 0);

end

D_cat = [D_cat, D];
Y_cat = [Y_cat, Y];

% figure
% subplot(121), visualizeDictionary(D0), title('Initial Dictionary')
% subplot(122), visualizeDictionary(D) , title('Trained Dictionary')
% 

%%

figure
imagesc(X)

H = kron(diag(ones(2,1)), ones(size(Y,2), 1))';

Y0 = Y_cat;
D0 = D_cat;

X0 = zeros(size(D0, 2), size(Y0, 2));

X0 = sparseCode(Y0, X0, D0, 20, 10);

W0 = H*X0'*inv(X0*X0'+eye(size(X0*X0')));

%%
gamma = 0.8;

D = [D0; sqrt(gamma)*W0];
Y = [Y0; sqrt(gamma)*H];
X = zeros(size(D,2), size(Y,2));

niter_learn = 20;
niter_coeff = 5;
niter_dict = 5;

for iter = 1:niter_learn
    fprintf('Dictionary Learning Iteration No. %d\n', iter);

    %%%%%%%%%%%%%%%% coefficient calculation %%%%%%%%%%%%%%%%%%%%%%%
    X = sparseCode(Y, X, D, T0, niter_coeff, 'Verbose', 0, 'StepSize', 10000);
    
   
    %%%%%%%%%%%%%%%% dictionary update %%%%%%%%%%%%%%%%%%%%%%%%%%%
    [D, X] = updateDictionary(Y, X, D, 'ksvd', 'nIter', niter_dict, 'Verbose', 0);

end

%%

D_final = (D(1:n,:))./(sqrt(sum(abs(D(1:n,:).^2),1)));
W_final = (D(n+1:end,:))./(sqrt(sum(abs(D(1:n,:).^2),1)));


%%

residual = (W_final*X);
[c, i] = max(abs(residual))


label=zeros(size(residual));
label(i)=residual(i);


figure
imagesc(i)

% visualizeDictionary(Y)
% figure
% visualizeDictionary(repmat(i, 1024,1))

% imagesc(repmat(i, 1024,1))

%%

X0 = zeros(size(D_final, 2), size(Y_cat, 2));

X = sparseCode(Y_cat, X0, D_final, 1, 10, 'Verbose', 1);




%%
PA = reshape((D*X), [blockSize blockSize size(Y, 2)]);
PA = PA - repmat( mean(mean(PA)), [blockSize blockSize] );
PA = PA + reshape(repmat( meanY, [blockSize^2 1]), [blockSize blockSize  size(Y, 2)]);

W = zeros(imH, imW);
denoisedImage = zeros(imH, imW);

for i=1:size(Y, 2)
    x = Xp(:,:,i);
    y = Yp(:,:,i);
    
    denoisedImage(x+(y-1)*imH) = denoisedImage(x+(y-1)*imH) + PA(:,:,i);
    W(x+(y-1)*imH) = W(x+(y-1)*imH) + 1;
end

denoisedImage = denoisedImage ./ W;


figure,
subplot(121), imagesc(image), title('Noisy image'), axis image
subplot(122), imagesc(denoisedImage), title('Denoised image'), axis image


%%


























