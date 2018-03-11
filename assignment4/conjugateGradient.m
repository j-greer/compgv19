function [xMin, nIter, resV, info] = conjugateGradient(A, b, tol, maxIter, M, x0, xtrue)
% CONJUGATE GRADIENT Solves the implicitely symmetric preconditioned CG
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
%   - xs: iterate history
%   Notice that the left preconditioned NE (with M inner product) and
%   the right preconditioned NE (with M^-1 inner product) produce the same algorithm, 
%   hence only one version.
%
% Copyright (C) 2017 Marta M. Betcke, Kiko Rul·lan

% A and M can be either matrices or function handlers
if ~strcmp(class(A), 'function_handle')
  afun = @(x) A*x;
else
  afun = A;
end
if ~strcmp(class(M), 'function_handle')
  mfun = @(x) M\x;
else
  mfun = M;
end

%===================== YOUR CODE HERE =============================
%initalize
r0 = A*x0 - b;
y0 = M(A)\r0;
p0 = -y0;
k = 0;
resV = [r0];
info.xs = [x0];

r_k = r0;
y_k = y0;
x_k = x0;
p_k = p0;

%check for stopping condition
while (abs(norm(r_k)) > tol) && (k < maxIter)
    alpha_k = r_k'*y_k/(p_k'*A*p_k);
    x_k_plus_1 = x_k + alpha_k*p_k;
    r_k_plus_1 = r_k + alpha_k*A*p_k;
    
    y_k_plus_1 = M(A)\r_k_plus_1;
    beta_k_plus_1 = r_k'*y_k_plus_1/(r_k'*y_k);
    p_k_plus_1 = -y_k_plus_1 + beta_k_plus_1*p_k;
    k = k + 1;
    
    %update variables
    p_k = p_k_plus_1;
    x_k = x_k_plus_1;
    r_k = r_k_plus_1;
    y_k = y_k_plus_1;
    
    resV = [resV r_k];
    info.xs = [info.xs x_k];
end

xMin = x_k;
nIter = k;

%==================================================================
