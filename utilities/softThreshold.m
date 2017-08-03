function [ out ] = softThreshold( in, th )
%softThreshold Summary of this function goes here
%   Detailed explanation goes here

    out = sign(in) .* max(abs(in)-th,0);
    
end

