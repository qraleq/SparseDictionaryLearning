function [ out ] = hardThreshold( in, th )
%hardThreshold Summary of this function goes here
%   Detailed explanation goes here

    out =  in .* (abs(in)>th);
    
end

