function [ out ] = normalizeColumns( in )
%normalizeColumns Summary of this function goes here
%   Detailed explanation goes here

out = in./(sqrt(sum(in.^2)));

end

