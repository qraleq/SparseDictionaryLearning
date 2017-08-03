close all
% clearvars
clc

% load image for patch extraction
im = im2double(imresize(imread('lena.png'), 0.5));
n0 = size(im,1);

% im = im + 0.1*randn(size(im));

% mask = ones(16,16);
% mask = abs(randn(16,16))>0.3;
mask = abs(randn(size(im)))>0.3;
% M = diag(mask(:));

% M(M>0.3) =1 ;


% im = repmat(mask, size(im)/16).*im;
im = mask .* im;

% size of extracted patch - 8x8 JPEG like
w = 16;

% number of image patches in set Y
m = 5000;

% length of signal y
n = w^2;

% desired sparsity
k = 10;

% number of atoms in dictionary
p = 128;

% overlap
q = 1 ;

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

M = mask(Xp+(Yp-1)*n0);
M = reshape(M, [n, m]);

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

%%

niter_coeff = 1;
D = overcompleteDCTdictionary(n, 300);
% D = D0;
% load D4.mat

X = zeros(size(D,2),size(Y,2));
E0 = [];

sigma = 0.000001;
lambda = 1.5 * sigma;

tau = 1.9/norm(D*D');
E = [];
th=tau*lambda;

step=5000;

for jj = 1:step:size(Y,2)
    jj
    jumpSize=min(jj+step-1,size(Y,2));
    X_tmp = zeros(size(D,2),1);
    
    for i = 1:niter_coeff

        %                 Y(:,jj:jumpSize)
        i
        
        for kk = jj:jumpSize
            
%             kk
            R = M(:,kk).*D*X_tmp-Y(:,kk);
            %         E(end+1,:) = sum(R.^2);
            
            %                     X = hardThresh(X-tau*D'*R, k);
            
            th = tau*lambda;
            X_tmp = softThresh(X_tmp-tau*(M(:,kk).*D)'*R, th');
            
            
            
            %         X = mexOMP(Y(:,jj:jumpSize),D,param);
            %         X = OMPerr(D,Y,sigma);
            
            X(:,kk)=X_tmp;
        end
        
        %             X = wthresh(X-tau*D'*R, 'h', tau*lambda);
    end
end





%%

% figure
% dictVisual = col2im(D, [w,w], size(D), 'distinct');
%
% imagesc(dictVisual), axis image
% % pcolor(randn(15,21))
% xticks(0.5:w:size(dictVisual,2))
% yticks(0.5:w:size(dictVisual,1))
% grid on
%
% ax = gca;
% ax.GridColor = 'black';
% ax.GridAlpha = 1;
% set(gca,'LineWidth', 2);


%%

PA = reshape(D*X, [w w m]);
PA = PA - repmat( mean(mean(PA)), [w w] );
PA = PA + reshape(repmat( a, [w^2 1] ), [w w m]);

n=n0;
W = zeros(n,n);
M1 = zeros(n,n);
% M1= zeros(size(im))
% W = zeros(size(im))


for i=1:m
    x = Xp(:,:,i); y = Yp(:,:,i);
    M1(x+(y-1)*n) = M1(x+(y-1)*n) + PA(:,:,i);
    W(x+(y-1)*n) = W(x+(y-1)*n) + 1;
    %     M1(Xp+(Yp-1)*n) = M1(Xp+(Yp-1)*n) + PA(:,:,i);
    %     W(Xp+(Yp-1)*n) = W(Xp+(Yp-1)*n) + 1;
end
M1 = M1 ./ W;



figure
% imageplot(clamp(M), ['Noisy, SNR=' num2str(snr(M0,M),4) 'dB'], 1, 2, 1);
imagesc(M1)