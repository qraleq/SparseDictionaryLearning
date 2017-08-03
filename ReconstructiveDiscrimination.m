close all
clear all
clc
%%

D0 = [];
Y = [];

directoryPath = 'Y:\Projects\MATLAB Projects\Dictionary Learning\data\textures\';
fileExtension = '.tiff';

files = dir(fullfile(directoryPath, strcat('*', fileExtension)));



for i=1:1
    
%     fileName = files(i).name;

    fileName = 'Y:\Projects\MATLAB Projects\Dictionary Learning\data\textures\1.1.02.tiff';
    
    image=im2double(imread((fileName)));
    image = imresize(image,0.5);
%     image=im2double(imread(strcat(directoryPath, fileName)));
    
    if(size(image,3) > 1)
        image = rgb2gray(image);
    end
    
%     if(p.Results.Plot)
%         figure(1)
%         imagesc(image)
%         drawnow
%     end
    
%     images(:,:,i)=(image);
%     disp(['File No: ' num2str(i, '%02d')]);
    
    
    %%%%%%%%%%%%%
    
    % size of extracted square (w*w) patch
    w = 16;
    
    % number of image patches in set Y
    m = 5000;
    
    % length of signal y (vectorized image patch)
    n = w^2;
    
    % desired sparsity (number of non-zero elements in sparse representation vector)
    K = 10;
    
    % number of atoms in dictionary D
    p = 500;
       
    [N1, N2] = size(image);
    
    % number of randomly selected image patches
    q = 3*m;
    
    % select q random locations in image (upper left block corners)
    x = floor(rand(1,1,q)*(N1-w))+1;
    y = floor(rand(1,1,q)*(N2-w))+1;
    
    % create rectangular mesh wxw
    [dY,dX] = meshgrid(0:w-1,0:w-1);
    
    % generate matrices containting block locations
    Xp = repmat(dX, [1 1 q]) + repmat(x, [w w 1]);
    Yp = repmat(dY, [1 1 q]) + repmat(y, [w w 1]);
    
    % extract and vectorize blocks
    Y_part = image(Xp+(Yp-1)*N1);
    Y_part = reshape(Y_part, [n, q]);
    
    % substract mean value from the blocks
    Y_part = Y_part - repmat(mean(Y_part), [n,1]);
    
    Y_part = unique(Y_part', 'rows')';
    
    % choose m highest energy blocks
    [~, idx] = sort(sum(Y_part.^2), 'descend');
    Y_part = Y_part(:, idx(1:m));
    
    % randomly select p blocks (p atoms in the dictionary D)
    sel = randperm(m);
    sel = sel(1:p);
    
    % normalize columns of Y (normalized image patches)
    D0_part = normc(Y_part(:,sel));
    
    D0 = [D0, D0_part];
    Y = [Y, Y_part];
    
end


%%

% param.K = 100;  % learns a dictionary with 100 elements
param.lambda = 0.05;
param.numThreads = -1; % number of threads
param.batchsize = 1024;
param.verbose = true;
param.iter = 20;  % let us see what happens after 1000 iterations.
param.mode = 5;
param.D = D0;

D = mexTrainDL(Y, param);

%%


% load image for patch extraction
filePath = 'Y:\Projects\MATLAB Projects\Dictionary Learning\data\textures\1234.tiff';

image = im2double(imresize(imread(filePath), 0.5));

if(size(image,3) > 1)
    image = rgb2gray(image);
end

[N1, N2] = size(image);

% size of extracted patch - 8x8 JPEG like

% number of image patches in set Y
m = 5000;

% length of signal y
n = w^2;

% desired sparsity
k = 256;

% number of atoms in dictionary
p = 128;

% overlap
q = 1;

[y, x] = meshgrid(1:q:N2-w/2, 1:q:N1-w/2);
[dY,dX] = meshgrid(0:w-1,0:w-1);

m = size(x(:),1);
% m=256;

Xp = repmat(dX,[1 1 m]) + repmat( reshape(x(:),[1 1 m]), [w w 1]);
Yp = repmat(dY,[1 1 m]) + repmat( reshape(y(:),[1 1 m]), [w w 1]);

Xp(Xp>N1) = 2*N1-Xp(Xp>N1);
Yp(Yp>N2) = 2*N2-Yp(Yp>N2);

Y = image(Xp+(Yp-1)*N1);
Y = reshape(Y, [n, m]);

a = mean(Y);
Y = Y - repmat(mean(Y), [n,1]);


select = @(A,k)repmat(A(k,:), [size(A,1) 1]);
hardThresh = @(X,k)X .* (abs(X) >= select(sort(abs(X), 'descend'),k));
softThresh = @(X,th)sign(X).*max(abs(X)-th,0);

%%

niter_coeff = 20;

% load D.mat

% D=D3;

X = zeros(size(D,2),size(Y,2));
E0 = [];

sigma = 0;
lambda = 1.5 * sigma;

tau = 1.9/norm(D*D');
E = [];
th=tau*lambda;

step = 2000;

for jj = 1:step:size(Y,2)
    jj
    jumpSize=min(jj+step-1,size(Y,2));
    
    X_tmp = zeros(size(D,2),size(Y(:,jj:jumpSize),2));
    
    for i = 1:niter_coeff
        
        
        
        %                 Y(:,jj:jumpSize)
        i
        R(:,jj:jumpSize) = D*X_tmp-Y(:,jj:jumpSize);
        %         E(end+1,:) = sum(R.^2);
        
        X_tmp = hardThresh(X_tmp-tau*D'*R(:,jj:jumpSize) , 5);
        
%         th = tau*lambda;
%         X_tmp = softThresh(X_tmp-tau*D'*R, th');
        
        
%         param.eps=sigma;
%         param.lambda=0.000000001;
%         param.mode = 2;
        
%                         X_tmp = mexLasso(Y(:,jj:jumpSize),D,param);

%                 X_tmp = mexOMP(Y(:,jj:jumpSize),D,param);
%                 X_tmp = OMPerr(D,Y(:,jj:jumpSize),sigma);
        
        X(:,jj:jumpSize)=X_tmp;
        
        
        %             X = wthresh(X-tau*D'*R, 'h', tau*lambda);
    end
end

%%
% R = D*X - Y;

% sum(sum(R.^2))
% 
% Y_rec = D*X;

Y_rec = R;
% 
PA = reshape(Y_rec, [w w m]);

% PA = PA - repmat( mean(mean(PA)), [w w] );
% PA = PA + reshape(repmat( a, [w^2 1] ), [w w m]);


W = zeros(N1,N2);
M1 = zeros(N1,N2);


for i=1:m
    x = Xp(:,:,i); y = Yp(:,:,i);
    
    M1(x+(y-1)*N1) = M1(x+(y-1)*N1) + PA(:,:,i);
    
    W(x+(y-1)*N1) = W(x+(y-1)*N1) + 1;
end

M1 = M1 ./ W;


figure
imagesc(M1.^2)

nanmean(M1(:).^2)

