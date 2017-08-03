%% LEARNING A LINEAR CLASSIFIER WITH KSVD
close all
clearvars
clc

%% INITIALIZATION

addpath('utilities')
addpath('data')

% size of extracted square (w*w) patch
w = 16;

% length of signal y (vectorized image patch)
n = w^2;

Y_cat = [];
D_cat = [];

% FUNCTIONS
select = @(A,k)repmat(A(k,:), [size(A,1) 1]);
strictThresh = @(X,k) X .* (abs(X) >= select(sort(abs(X), 'descend'),k));

softThresh = @(X,th) sign(X) .* max(abs(X)-th,0);
hardThresh = @(X,th) X .* (abs(X)>th);

normCols = @(X) X./ repmat(sqrt(sum(X.^2)), [n, 1]);


%%
% desired sparsity (number of non-zero elements in sparse representation vector)
K = 4;

% number of atoms in dictionary D
p = 64;

% load image for patch extraction
imagePath = '.\data\textures\1.1.05.tiff';
image = im2double(imresize(imread(imagePath), 0.5));

% add additive noise noise
% sigma = 0.1;
% im = im + sigma*randn(size(im));

[N1, N2] = size(image);


% overlap
q = w;

[y, x] = meshgrid(1:q:N2-w/2, 1:q:N1-w/2);
[dY,dX] = meshgrid(0:w-1,0:w-1);

m = size(x(:),1);
% m=256;

Xp = repmat(dX,[1 1 m]) + repmat( reshape(x(:),[1 1 m]), [w w 1]);
Yp = repmat(dY,[1 1 m]) + repmat( reshape(y(:),[1 1 m]), [w w 1]);

Xp(Xp>N1) = 2*N1-Xp(Xp>N1);
Yp(Yp>N2) = 2*N2-Yp(Yp>N2);

Y = image(Xp+(Yp-1)*N1);


h = fspecial('gaussian', 16, 5);
Y = imfilter(Y, h, 'replicate');

Y = reshape(Y, [n, m]);

a = mean(Y);
Y = Y - repmat(mean(Y), [n,1]);


% number of randomly selected image patches
m = 3000;
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
Y = image(Xp+(Yp-1)*N1);
Y = reshape(Y, [n, q]);

% substract mean value from the blocks
Y = Y - repmat(mean(Y), [n,1]);

% choose m highest energy blocks
[~, idx] = sort(sum(Y.^2), 'descend');
Y = Y(:, idx(1:m));

% randomly select p blocks (p atoms in the dictionary D)
sel = randperm(m);
sel = sel(1:p);

% normalize columns of Y (normalized image patches)
D0 = normCols(Y(:,sel));

% Y_cat = [Y_cat, Y];
% D_cat = [D_cat, D0];


% DICTIONARY LEARNING
% perform dictionary learning by iteratively repeating coefficient
% calculation and dictionary update steps

niter_learn = 5;
niter_dict = 5;
niter_coeff = 5;
% K = 10;

X = zeros(size(D0,2), size(Y,2));
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
        
        
%                 X = softThresh(X-tau1*D'*R, th);
        %             X = hardThresh(X-tau1*D'*R, th);
        X = strictThresh(X-tau1*D'*R, K);
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
    
    for kk=1:size(D,2)
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

Y_cat = [Y_cat, Y];
D_cat = [D_cat, D0];


figure,
hold on
plot(1:2*niter_learn, E0);
plot(1:2:2*niter_learn, E0(1:2:2*niter_learn), '*');
plot(2:2:2*niter_learn, E0(2:2:2*niter_learn), 'o');
axis tight;
legend('|Y-DX|^2', 'After coefficient update', 'After dictionary update');



%%

H = kron(diag(ones(5,1)), ones(size(Y,2),1))';
Y = Y_cat;
D0 = D_cat;
D = D0;

X = zeros(size(D,2),size(Y,2));
E0 = [];

sigma = 0.1;
lambda = 1.5 * sigma;

tau = 1.5/norm(D*D');
E = [];
th=tau*lambda;

step = 1000;

for jj = 1:step:size(Y,2)
    jj
    
    jumpSize = min(jj+step-1,size(Y,2));
    X_tmp = zeros(size(D,2),size(Y(:,jj:jumpSize),2));
    
    for i = 1:niter_coeff
                
        i
        R = D*X_tmp-Y(:,jj:jumpSize);
       
        X_tmp = strictThresh(X_tmp-tau*D'*R, K);
        
        
        X(:,jj:jumpSize)=X_tmp;
        
        
    end
end


lambda=1;
W0 = H*X'*inv(X*X'+lambda*eye(size(X*X')))
D0=D_cat;
Y0=Y_cat;

%% DICTIONARY LEARNING
% perform dictionary learning by iteratively repeating coefficient
% calculation and dictionary update steps

niter_learn = 5;
niter_dict = 20;
niter_coeff = 5;
% K = 5;

E0 = [];

D = [D0; 10*W0];
Y = [Y0; 10*H];
X = zeros(size(D,2), size(Y,2));

sigma = 0.1;
lambda = 1.5 * sigma;

for iter = 1:niter_learn
    fprintf('Dictionary Learning Iteration No. %d\n', iter);
    
    %%%%%%%%%%%%%%%% coefficient calculation %%%%%%%%%%%%%%%%%%%%%%%
    tau1 = 1.9/norm(D*D');
    E = [];
    
    th = tau1*lambda;
    
    for i = 1:niter_coeff
%         fprintf('Coefficient Calculation Iteration No. %d\n', i);
        
        % calculate residual of signal estimation
        R = D*X - Y;
        
        % calculate energy for energy decay visualization
        E(end+1,:) = sum(R.^2);
        
        
%                 X = softThresh(X-tau1*D'*R, th);
        %             X = hardThresh(X-tau1*D'*R, th);
        X = strictThresh(X-tau1*D'*R, K);
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
    
    for kk=1:size(D,2)
%         fprintf('K-SVD Current Column No. %d\n', kk);
        
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
%%

D_final = (D(1:n,:))./(sqrt(sum(abs(D(1:n,:).^2),1)));
W_final = (D(n+1:end,:))./(sqrt(sum(abs(D(1:n,:).^2),1).^2));

%%
% 
% 
% A = [1 2 3; 4 5 6; 7 8 9];
% 
% l2norm_col=sqrt(sum(A.*A,1))
% A = A./repmat(l2norm_col, size(A,1), 1)
% 
% 
% A = A./(sqrt(sum(abs(A).^2,1)))

%%
X_test = X;
% X_test(:,769:end)=X_test(:,1:256);
% X_test(:,257:257+255)=X_test(:,1:256);
% X_test(:,1:256)=X_test(:,257:257+255);
% X_test(:,1:256)=X_test(:,257+255:257+255+255);

lambda=1;

% W = H*X_test'*inv(X_test*X_test'+lambda*eye(size(X_test*X_test')));

% W = n(W)

% W = H*X'*inv(X*X'+lambda*eye(size(X*X')))

% label = abs(W_final*X_test);
label = abs(W0*X_test);

% imagesc(label)

[maximum, idx]=max(label,[], 1)

figure
imagesc(idx)


%%

% size of extracted square (w*w) patch

% length of signal y (vectorized image patch)
n = w^2;

% desired sparsity (number of non-zero elements in sparse representation vector)
% K = 1;

% number of atoms in dictionary D
p = 100;

% load image for patch extraction
imagePath = '.\data\textures\1.1.01.tiff';
image = im2double(imresize(imread(imagePath), 0.5));

% add additive noise noise
% sigma = 0.1;
% im = im + sigma*randn(size(im));

[N1, N2] = size(image);

% overlap
q = w;

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


niter_coeff = 100;
% D0 = overcompleteDCTdictionary(n, 800);
D = D_final;
% load D.mat

X = zeros(size(D,2),size(Y,2));
E0 = [];

sigma = 0.1;
lambda = 1.5 * sigma;

tau = 1.5/norm(D*D');
E = [];
th=tau*lambda;

step = 1000;

for jj = 1:step:size(Y,2)
    jj
    
    jumpSize = min(jj+step-1,size(Y,2));
    X_tmp = zeros(size(D,2),size(Y(:,jj:jumpSize),2));
    
    for i = 1:niter_coeff
        
        
        
        %                 Y(:,jj:jumpSize)
        i
        R = D*X_tmp-Y(:,jj:jumpSize);
        %         E(end+1,:) = sum(R.^2);
        
        X_tmp = strictThresh(X_tmp-tau*D'*R, K);
        
%         th = tau*lambda;
%         X_tmp = softThresh(X_tmp-tau*D'*R, th');
        
        
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
nVis=10;
visualization=Y;
figure
dictVisual = col2im(visualization(:,1:w*nVis), [w,w], size(visualization(:,1:w*nVis)), 'distinct');

imagesc(dictVisual), axis image
xticks(0.5:w:size(dictVisual,2))
yticks(0.5:w:size(dictVisual,1))
grid on

ax = gca;
ax.GridColor = 'black';
ax.GridAlpha = 1;
set(gca,'LineWidth', 2);


%%

idx_compl = repmat(idx, n,1);

