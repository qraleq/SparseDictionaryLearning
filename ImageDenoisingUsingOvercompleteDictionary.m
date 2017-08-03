close all
% clearvars
clc
%% INITIALIZATION

addpath('utilities')
addpath('data')

% load image for patch extraction
im = im2double(imresize(imread('barb.png'), 0.5));
n0 = size(im,1);

im = im + 0.1*randn(size(im));

mask = ones(16,16);
% mask = abs(randn(16,16))>0.3;
% M = diag(mask(:));
% im = repmat(mask, size(im)/16).*im;

% size of extracted patch - 8x8 JPEG like
w = 16;

% number of image patches in set Y
m = 5000;

% length of signal y
n = w^2;

% desired sparsity
k = 20;

% number of atoms in dictionary
p = 128;

% overlap
q = 1;

[y, x] = meshgrid(1:q:n0-w/2, 1:q:n0-w/2);
[dY,dX] = meshgrid(0:w-1,0:w-1);

m = size(x(:),1);
% m=256;

Xp = repmat(dX,[1 1 m]) + repmat( reshape(x(:),[1 1 m]), [w w 1]);
Yp = repmat(dY,[1 1 m]) + repmat( reshape(y(:),[1 1 m]), [w w 1]);

Xp(Xp>n0) = 2*n0-Xp(Xp>n0);
Yp(Yp>n0) = 2*n0-Yp(Yp>n0);

Y = im(Xp+(Yp-1)*n0);
Y = reshape(Y, [n, m]);

a = mean(Y);
Y = Y - repmat(mean(Y), [n,1]);
% D0 = normc(Y);


% figure
% dictVisual = col2im(D0, [w,w], size(D0), 'distinct');
%
% imagesc(dictVisual), axis image
% xticks(0.5:w:size(dictVisual,2))
% yticks(0.5:w:size(dictVisual,1))
% grid on
%
% ax = gca;
% ax.GridColor = 'black';
% ax.GridAlpha = 1;
% set(gca,'LineWidth', 2);

select = @(A,k)repmat(A(k,:), [size(A,1) 1]);
hardThresh = @(X,k)X .* (abs(X) >= select(sort(abs(X), 'descend'),k));
softThresh = @(X,th)sign(X).*max(abs(X)-th,0);

%% DICTIONARY LEARNING



niter_coeff = 5;
D0 = overcompleteDCTdictionary(n, 800);
D = D0;
% load D.mat

X = zeros(size(D,2),size(Y,2));
E0 = [];

sigma = 0.1;
lambda = 1.5 * sigma;

tau = 1.9/norm(D*D');
E = [];
th=tau*lambda;

step = 500;

for jj = 1:step:size(Y,2)
    jj
    jumpSize=min(jj+step-1,size(Y,2));
    X_tmp = zeros(size(D,2),size(Y(:,jj:jumpSize),2));
    
    for i = 1:niter_coeff
        
        
        
        %                 Y(:,jj:jumpSize)
        i
        R = D*X_tmp-Y(:,jj:jumpSize);
        %         E(end+1,:) = sum(R.^2);
        
%                             X_tmp = hardThresh(X_tmp-tau*D'*R, k);
        
%         th = tau*lambda;
        X_tmp = softThresh(X_tmp-tau*D'*R, th');
        
        
%         param.eps=sigma;
%         param.lambda=0.1;
%         param.mode = 2;
        
%                         X_tmp = mexLasso(Y(:,jj:jumpSize),D,param);

%                 X_tmp = mexOMP(Y(:,jj:jumpSize),D,param);
%                 X_tmp = OMPerr(D,Y(:,jj:jumpSize),sigma);
        
        X(:,jj:jumpSize)=X_tmp;
        
        
        %             X = wthresh(X-tau*D'*R, 'h', tau*lambda);
    end
end

%%
weights = repmat(sum(X~=0),256,1);
% weights = repmat(sum(X),256,1);

% weights = ones(size(weights));

PA = reshape((weights./100).*(D*X), [w w m]);
PA = PA - repmat( mean(mean(PA)), [w w] );
PA = PA + reshape(repmat( a, [w^2 1] ), [w w m]);

n=n0;
W = zeros(n,n);
M1 = zeros(n,n);


for i=1:m
    x = Xp(:,:,i); y = Yp(:,:,i);
    M1(x+(y-1)*n) = M1(x+(y-1)*n) + PA(:,:,i);
    W(x+(y-1)*n) = W(x+(y-1)*n) + 1;
end

M1 = M1 ./ W;


figure
imagesc(M1)