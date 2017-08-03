function [ X ] = sparseCode( Y, X, D, T0, nIter, varargin )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;

p.addRequired('Y', @ismatrix);
p.addRequired('X', @ismatrix);
p.addRequired('D', @ismatrix);
p.addRequired('T0', @(x) isnumeric(x) && x>0);
p.addRequired('nIter', @(x) isnumeric(x) && x>0);
p.addParameter('StepSize', 0, @isnumeric);
p.addParameter('Plot', 0, @isnumeric);
p.addParameter('Verbose', 0, @isnumeric);

p.parse(Y, X, D, T0, nIter, varargin{:})


if(p.Results.StepSize == 0)
    step = size(Y,2);
else
    step = p.Results.StepSize;
end

% initialization
% X = zeros(size(D, 2), size(Y, 2));

% gradient descent step
tau = 1.6/norm(D*D');

E = [];

sigma = 0.1;
% lambda controls sparsity of the coefficients
% l1 regularization is similar to soft thresholding and then usual
% 1.5*sigma value is used as lambda
lambda = 1.5 * sigma;

% soft threshold
th = lambda*tau;

% step = 10000;

for jj = 1:step:size(Y,2)
    if(p.Results.Verbose)
        fprintf('Sparse Coding Col. No. %d - %d/%d\n', jj, min(size(Y,2), jj + min(step, size(Y,2))), size(Y,2));
    end
    
    jumpSize=min(jj+step-1,size(Y,2));
    X_tmp = zeros(size(D,2),size(Y(:,jj:jumpSize),2));
    R = [];
    
    for i = 1:nIter
        R = D*X_tmp-Y(:,jj:jumpSize);
        
                X_tmp = strictThreshold(X_tmp-tau*D'*R, T0);
        
        %         th = tau*lambda;
%         X_tmp = softThreshold(X_tmp-tau*D'*R, th');
        
        X(:,jj:jumpSize)=X_tmp;
        
    end
end


if(p.Results.Plot)
    sel = 1:10;
    figure,
    plot(log10(E(1:end,sel) - repmat(min(E(:,sel),[],1),[nIter 1])));
    axis tight;
    title('$$log_{10}(J(x_j) - J(x_j^*))$$', 'Interpreter', 'latex');
end

end

