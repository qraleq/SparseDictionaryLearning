function [ out ] = strictThreshold( in, k )
%strictThreshold Summary of this function goes here
%   Detailed explanation goes here

select = @(A,k)repmat(A(k,:), [size(A,1) 1]);
out = in .* (abs(in) >= select(sort(abs(in), 'descend'), k));

end

