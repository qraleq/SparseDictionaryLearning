function [ D, X, E0 ] = learnDictionary( Y, D, T0, varargin )
%learnDictionary Summary of this function goes here
%   Detailed explanation goes here

p = inputParser;

p.addRequired('Y', @ismatrix);
p.addRequired('D', @ismatrix);
p.addRequired('T0', @isnumeric);
p.addParameter('nIterLearn', 10, @isnumeric);
p.addParameter('nIterDict', 10, @isnumeric);
p.addParameter('nIterCoeff', 10, @isnumeric);
p.addParameter('modeDict', 'ksvd', @isstr);
p.addParameter('modeCoeff', 'grad', @isstr);
p.addParameter('Plot', 0, @isnumeric);
p.addParameter('Verbose', 0, @isnumeric);

p.parse(Y, D, T0, varargin{:});

X = zeros(size(D, 2), size(Y, 2));
E0 = [];

for iter = 1:p.Results.nIterLearn
    fprintf('Dictionary Learning Iteration No. %d\n', iter);
    
    %%%%%%%%%%%%%%%% coefficient calculation %%%%%%%%%%%%%%%%%%%%%%%
    X = sparseCode(Y, D, T0, p.Results.nIterCoeff);
    
    E0(end+1) = norm(Y-D*X, 'fro')^2;
    
    %%%%%%%%%%%%%%%% dictionary update %%%%%%%%%%%%%%%%%%%%%%%%%%%
    [D, X] = updateDictionary(Y, X, D, p.Results.modeDict, 'nIter', p.Results.nIterDict);
    
    E0(end+1) = norm(Y-D*X, 'fro')^2;
end

if(p.Results.Plot)
    figure,
    hold on
    plot(1:2*p.Results.nIterLearn, E0);
    plot(1:2:2*p.Results.nIterLearn, E0(1:2:2*p.Results.nIterLearn), '*');
    plot(2:2:2*p.Results.nIterLearn, E0(2:2:2*p.Results.nIterLearn), 'o');
    axis tight;
    legend('|Y-DX|^2', 'After coefficient update', 'After dictionary update');
end


end

