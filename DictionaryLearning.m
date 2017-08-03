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
image = im2double(imresize(imread(imagePath), 0.5));

% add additive noise noise
% sigma = 0.1;
% im = im + sigma*randn(size(im));

[imH, imW] = size(image, blockSize);

%% EXTRACT IMAGE PATCHES & INITIALIZE DICTIONARY D0 & PLOT DICTIONARY

Y = extractImagePatches(image, blockSize);

%%

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
Y = image(Xp+(Yp-1)*imH);
Y = reshape(Y, [n, q]);

% substract mean value from the blocks
Y = Y - repmat(mean(Y), [n,1]);

% choose m highest energy blocks
[~, idx] = sort(sum(Y.^2), 'descend');
Y = Y(:, idx(1:N));

% randomly select p blocks (p atoms in the dictionary D)
sel = randperm(N);
sel = sel(1:K);

% normalize columns of Y (normalized image patches)
D0 = normalizeColumns(Y(:,sel));

% ALTERNATIVE: generate overcomplete DCT dictionary
% D0 = overcompleteDCTdictionary(n, p);

nVis = 16;
figure
dictVisual = col2im(D0(:,1:blockSize*nVis), [blockSize,blockSize], size(D0(:,1:blockSize*nVis)), 'distinct');

imagesc(dictVisual), axis image
xticks(0.5:blockSize:size(dictVisual,2))
yticks(0.5:blockSize:size(dictVisual,1))
grid on

ax = gca;
ax.GridColor = 'black';
ax.GridAlpha = 1;
set(gca,'LineWidth', 2);



%% CALCULATE COEFFICIENTS X

% number of iterations for sparse coefficients update
niter_coeff = 20;

% initialization
D = D0;
X = zeros(K, N);

% gradient descent step
tau = 1.95/norm(D*D');

E = [];

sigma = 0.1;
% lambda controls sparsity of the coefficients
% l1 regularization is similar to soft thresholding and then usual
% 1.5*sigma value is used as lambda
lambda = 1.5 * sigma;

% soft threshold
th = lambda*tau;

for i = 1:niter_coeff
    % calculate residual of signal estimation
    R = D*X-Y;
    
    % calculate energy for energy decay visualization
    E(end+1,:) = sum(R.^2);
    
    %     X = softThreshold(X-tau*D'*R, th);
    X = strictThreshold(X-tau*D'*R, T0);
    %     X = wthresh(X-tau*D'*R, 's', th);
    
end

sel = 1:10;
figure,
plot(log10(E(1:end,sel) - repmat(min(E(:,sel),[],1),[niter_coeff 1])));
axis tight;
title('$$log_{10}(J(x_j) - J(x_j^*))$$', 'Interpreter', 'latex');


%% UPDATE DICTIONARY D

niter_dict = 20;
tau = 1.6/norm(X*X');
E = [];
D = D0;

% niter_dict iterations of projected gradient descent for updating
% dictionary
for i = 1:niter_dict
    R = D*X-Y;
    
    E(end+1) = sum(R(:).^2);
    % update dictionary using projected gradient descent with orthogonal
    % projector(unit norm normalized columns)
    D = normCols(D - tau*(D*X - Y)*X');
end

figure,
plot(log10(E(1:end/2)-min(E)));
axis tight;

%% DICTIONARY LEARNING
% perform dictionary learning by iteratively repeating coefficient
% calculation and dictionary update steps

niter_learn = 20;
niter_dict = 10;
niter_coeff = 10;

X = zeros(K,N);
E0 = [];

D = D0;

sigma = 0.1;
lambda = 1.5 * sigma;

for iter = 1:niter_learn
    %%%%%%%%%%%%%%%% coefficient calculation %%%%%%%%%%%%%%%%%%%%%%%
    tau1 = 1.6/norm(D*D');
    E = [];
    
    th = tau1*lambda;
    
    for i = 1:niter_coeff
        fprintf('Coefficient Calculation Iteration No. %d\n', i);
        
        % calculate residual of signal estimation
        R = D*X - Y;
        
        % calculate energy for energy decay visualization
        E(end+1,:) = sum(R.^2);
        
        
%                 X = softThreshold(X-tau1*D'*R, th);
        %             X = hardThreshold(X-tau1*D'*R, th);
        X = strictThreshold(X-tau1*D'*R, T0);
        %     X = wthresh(X-tau1*D'*R, 's', th);  
    end
    
    E0(end+1) = norm(Y-D*X, 'fro')^2;
    
    %%%%%%%%%%%%%%%% dictionary update %%%%%%%%%%%%%%%%%%%%%%%%%%%
    tau2 = 1/norm(X*X');
    
    E = [];
    
%     %     dictionary update by using projected gradient descent
%     for i = 1:niter_dict
%         fprintf('Dictionary Update Iteration No. %d\n', i);
%         
%         R = D*X-Y;
%         E(end+1) = sum(R(:).^2);
%         D = normCols(D - tau2*(D*X - Y)*X');
%     end
    
%     % dictionary update using MOD algorithm
%     % dictionary update is performed by minimizing Frobenius norm of R over all
%     % posible D and the solution is given by normalized pseudo-inverse
%     D = Y*pinv(X);
%     D = normCols(D);
    
    % dictionary update using K-SVD
    T = 1e-3;
    R = Y - D*X;
    E = [];
    
    for kk=1:K
        fprintf('K-SVD Current Column No. %d\n', kk);
        
        idx = find(abs(X(kk,:)) > T);
        if (~isempty(idx))
            Ri = R(:,idx) + D(:,kk)*X(kk,idx);
            
            [U,S,V] = svds(Ri,1,'L');
            
            D(:,kk) = U;
            X(kk,idx) = S*V';
            
            R(:,idx) = Ri - D(:,kk)*X(kk,idx);
            
            E(end+1) = sum(R(:).^2);
%             E
        end
    end
%     
%     
%     % dictionary update using approximate K-SVD
%     R = Y - D*X;
%     for kk=1:p
%         fprintf('K-SVD Current Column No. %d\n', kk);
% 
%         idx = find(abs(X(kk,:)) > T);
%         Ri = R(:,idx) + D(:,kk)*X(kk,idx);
%         dk = Ri * X(kk,idx)';
%         dk = dk/sqrt(dk'*dk);  % normalize
%         D(:,kk) = dk;
%         X(kk,idx) = dk'*Ri;
%         R(:,idx) = Ri - D(:,kk)*X(kk,idx);
%     end

    E0(end+1) = norm(Y-D*X, 'fro')^2;
end


figure,
hold on
plot(1:2*niter_learn, E0);
plot(1:2:2*niter_learn, E0(1:2:2*niter_learn), '*');
plot(2:2:2*niter_learn, E0(2:2:2*niter_learn), 'o');
axis tight;
legend('|Y-DX|^2', 'After coefficient update', 'After dictionary update');

%% MEX TRAIN

param.K = 100;  % learns a dictionary with 100 elements
param.lambda = 0.5;
param.numThreads = -1; % number of threads
param.batchsize = 512;
param.verbose = true;
param.iter = 50;  % let us see what happens after 1000 iterations.
param.mode = 5;
param.D = D0;

D = mexTrainDL(Y, param);


%%

figure
dictVisual = col2im(D(:,1:blockSize*nVis), [blockSize,blockSize], size(D(:,1:blockSize*nVis)), 'distinct');

imagesc(dictVisual), axis image
xticks(0.5:blockSize:size(dictVisual,2))
yticks(0.5:blockSize:size(dictVisual,1))
grid on

ax = gca;
ax.GridColor = 'black';
ax.GridAlpha = 1;
set(gca,'LineWidth', 2);


%%

[x_rec, r, coeff, iopt] = wmpalg('OMP', Y(:,1), D0, 'itermax', 100, 'maxerr', {'L2', 20});

iopt





