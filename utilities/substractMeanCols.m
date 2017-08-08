function [ out, meanValue] = substractMeanCols( in )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

meanValue = mean(in);
out = in - meanValue;

end

