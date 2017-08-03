function [ D, X ] = updateDictionary( Y, X, D, mode, varargin )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;

p.addRequired('Y', @ismatrix);
p.addRequired('X', @ismatrix);
p.addRequired('D', @ismatrix);
p.addParameter('nIter', 10, @(x) isnumeric(x) && x>0);
p.addRequired('mode', @isstr);
p.addParameter('Plot', 0, @isnumeric);
p.addParameter('Verbose', 0, @isnumeric);

p.parse(Y, X, D, mode,  varargin{:})


% nIter = 20;
tau = 1/norm(X*X');
E = [];
% D = D0;

if(strcmp(mode, 'grad'))
    % dictionary update by using projected gradient descent
    for i = 1:p.Results.nIter
        if(p.Results.Verbose)
            fprintf('Dictionary Update Iteration No. %d\n', i);
        end
        R = D*X-Y;
        E(end+1) = sum(R(:).^2);
        D = normalizeColumns(D - tau*(D*X - Y)*X');
    end
    
elseif(strcmp(mode, 'mod'))
    
    % dictionary update using MOD algorithm
    % dictionary update is performed by minimizing Frobenius norm of R over all
    % posible D and the solution is given by normalized pseudo-inverse
    D = Y*pinv(X);
    D = normalizeColumns(D);
    
    
elseif(strcmp(mode, 'ksvd'))
    
    % dictionary update using K-SVD
    T = 1e-3;
    R = Y - D*X;
    E = [];
    
    for kk=1:size(D,2)
        if(p.Results.Verbose)
            fprintf('K-SVD Current Column No. %d\n', kk);
        end
        idx = find(abs(X(kk,:)) > T);
        
        if (~isempty(idx))
            Ri = R(:,idx) + D(:,kk)*X(kk,idx);
            
            [U,S,V] = svds(Ri, 1, 'L');
            
            D(:,kk) = U;
            X(kk,idx) = S*V';
            
            R(:,idx) = Ri - D(:,kk)*X(kk,idx);
            
            E(end+1) = sum(R(:).^2);
        end
    end
    
elseif(strcmp(mode, 'aksvd'))
    
    % dictionary update using approximate K-SVD
    T = 1e-3;
    R = Y - D*X;
    E = [];
    
    for kk=1:size(D,2)
        if(p.Results.Verbose)
            fprintf('AK-SVD Current Column No. %d\n', kk);
        end
        
        idx = find(abs(X(kk,:)) > T);
        if (~isempty(idx))
            
            Ri = R(:,idx) + D(:,kk)*X(kk,idx);
            dk = Ri * X(kk,idx)';
            dk = dk/sqrt(dk'*dk);  % normalize
            D(:,kk) = dk;
            X(kk,idx) = dk'*Ri;
            R(:,idx) = Ri - D(:,kk)*X(kk,idx);
            
            E(end+1) = sum(R(:).^2);
            
        end
    end
    
end

if(p.Results.Plot)
    figure,
    plot(log10(E(1:end/2)-min(E)));
    axis tight;
end

end