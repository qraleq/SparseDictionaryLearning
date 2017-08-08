function [ D0 ] = initDictionaryFromPatches( n, K, patchesVectorized )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% choose K highest energy blocks
[~, idx] = sort(sum(patchesVectorized.^2), 'descend');
patchesVectorized = patchesVectorized(:, idx(1:size(patchesVectorized, 2)));

% randomly select p blocks (p atoms in the dictionary D)
% sel = randperm(size(patchesVectorized, 2));
% sel = sel(1:K);

sel = 1:K;

% normalize columns of Y (normalized image patches)
D0 = normalizeColumns(patchesVectorized(:,sel));

end

