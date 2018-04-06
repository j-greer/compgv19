function [xMin, nIter, resV, info] = totalVariation(A, b, lambda, tol, maxIter, P, x0)
% TOTAL VARIATION Solves the implicitely symmetric preconditioned CG
%function [x, flag, relres, iter, resvec, xs, V] = mcg(A, b, tol, maxit, M, x0)
% INPUTS
% A: symmetric matrix
% b: specifies the vector to solve Ax = b
% tol: tolerance for the residual
% maxIter: maximum number of iterations
% M: linear preconditioner matrix
% x0: initial iterate
%
% OUTPUTS
% xMin: solution of the system
% nIter: number of iterations taken
% resV: vector of residuals
% info: structure with information about the iteration
% - xs: iterate history
% Notice that the left preconditioned NE (with M inner product) and
% the right preconditioned NE (with M^-1 inner product) produce the same algorithm,
% hence only one version.
%
% Copyright (C) 2017 Marta M. Betcke, Kiko Rul·lan
if ~strcmp(class(A), 'function_handle')
    afun = @(x) A*x;
else
    afun = A;
end
% Define the derivative operators
finitediff = @(x,dir) x-circshift(x,dir);
DTwD = @(x,w) finitediff(w.*finitediff(x,[1 0]),[-1 0]) + finitediff(w.*finitediff(x,[0 1]),[0 -1]);

%Initialize:
x = x0;
j = 1;
nIter = 0;
resV = [];
info.xs = [];
while j <= maxIter
	w = (abs(finitediff(x,[1 0])).^2+abs(finitediff(x,[0 1])).^2);
    w = .5*lambda./sqrt(w + 1e-12*max(w(:)));
	M = @(x) afun(x)+DTwD(x,w);
    [x,tmp,dmp,~] = conjugateGradient(M,b,1e-3,maxIter,P,x);
    nIter = [nIter; nIter(end)+tmp];
    resV = [resV; dmp(2:end)];
    info.xs = [info.xs;reshape(x,[],1)];
    j = j + 1;
end
nIter = nIter(2:end);
xMin = x;
end
