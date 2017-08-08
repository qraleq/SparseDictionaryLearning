%IMAGE INPAINTING USING OVERCOMPLETE DICTIONARY
close all
clearvars
clc
%%


% load image for patch extraction
image = im2double(imresize(imread('lena.png'), 0.5));
[imH, imW] = size(image);

% im = im + 0.1*randn(size(im));

% generate random binary mask
mask = abs(randn(size(image)))>0.5;

% masking out image pixels
image = mask .* image;

% size of extracted patch
w = 16;

% number of image patches in set Y
N = 5000;

% length of signal y
n = w^2;

% desired sparsity
T0 = 10;

% number of atoms in dictionary
K = 128;

% overlap
q = 1;

[y, x] = meshgrid(1:q:imH-w/2, 1:q:imW-w/2);
[dY,dX] = meshgrid(0:w-1,0:w-1);

N = size(x(:),1);

Xp = repmat(dX,[1 1 N]) + repmat( reshape(x(:),[1 1 N]), [w w 1]);
Yp = repmat(dY,[1 1 N]) + repmat( reshape(y(:),[1 1 N]), [w w 1]);

Xp(Xp>imH) = 2*imH-Xp(Xp>imH);
Yp(Yp>imW) = 2*imW-Yp(Yp>imW);

Y = image(Xp+(Yp-1)*imH);
Y = reshape(Y, [n, N]);

M = mask(Xp+(Yp-1)*imH);
M = reshape(M, [n, N]);

[Y, meanY] = substractMeanCols(Y);


select = @(A,k)repmat(A(k,:), [size(A,1) 1]);
hardThresh = @(X,k)X .* (abs(X) >= select(sort(abs(X), 'descend'),k));
softThresh = @(X,th)sign(X).*max(abs(X)-th,0);

%%

niter_coeff = 1;
D = overcompleteDCTdictionary(n, 300);


X = zeros(size(D,2),size(Y,2));
E0 = [];

sigma = 0.000001;
lambda = 1.5 * sigma;

tau = 1.9/norm(D*D');
E = [];
th=tau*lambda;

step=1000;



for jj = 1:step:size(Y,2)
    jj
    
    jumpSize=min(jj+step-1,size(Y,2));
    X_tmp = zeros(size(D,2),1);
    
    for i = 1:niter_coeff
        
        i
        
        for kk = jj:jumpSize
            
            R = M(:,kk).*D*X_tmp-Y(:,kk);
            
%             X_tmp = hardThresh(X_tmp-tau*(M(:,kk).*D)'*R, T0);
%             X_tmp = OMP(M(:,kk).*D, Y(:,kk), T0);

%             param.lambda = th(1);
            param.L = 50;
%             X_tmp = mexLasso(Y(:,kk), M(:,kk).*D, param);
%             X_tmp = mexOMP(Y(:,kk), M(:,kk).*D, param);
            
            th = tau*lambda;
            X_tmp = softThresh(X_tmp-tau*(M(:,kk).*D)'*R, th');
            
            
            X(:,kk)=X_tmp;
        end
    end
end


%%

PA = reshape(D*X, [w w N]);
PA = PA - repmat( mean(mean(PA)), [w w]);
PA = PA + reshape(repmat( meanY, [w^2 1] ), [w w N]);

W = zeros(imH,imW);
M1 = zeros(imH, imW);


for i=1:N
    x = Xp(:,:,i); 
    y = Yp(:,:,i);
    
    M1(x+(y-1)*n) = M1(x+(y-1)*n) + PA(:,:,i);
    W(x+(y-1)*n) = W(x+(y-1)*n) + 1;
end

M1 = M1 ./ W;

figure
imagesc(M1)