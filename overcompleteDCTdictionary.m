function [ D ] = overcompleteDCTdictionary( M, K)
%overcompleteDCTdictionary Create overcomplete DCT MxK dictionary
%   Detailed explanation goes here

M=sqrt(M);
K=ceil(sqrt(K));

k=0:M/K:(M-M/K);
n=0:1:M-1;

E=cos(pi/M*k'*(n+0.5));

D=kron(E,E);
D=normc(D');

end

