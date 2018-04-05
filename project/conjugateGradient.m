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
if ~strcmp(class(M), 'function_handle')
    mfun = @(x) M\x;
else
    mfun = M;
end
n = size(reshape(b,[],1),1);
if nargin < 6
    x0 = zeros(size(b));
end
%Initialize:
x = x0./M; % Intial x
r0 = M.*b - M.*afun(M.*x); % residual: b - Ax
p0 = r0; % conjugate gradient
rnorm = norm(r0(:)); % residual norm |r|
bnorm = rnorm;
alpha = zeros(maxIter,1);
beta = zeros(maxIter,1);
resV = zeros(maxIter+1,1);
resV(1) = rnorm;
xs = zeros(n, maxIter+1);
V = zeros(n, maxIter+1);
xs(:,1) = x(:);
V(:,1) = r0(:);
j = 1;
while j <= maxIter && rnorm/bnorm >= tol %|r|/|b| > tol
    %Preconditioned Conjugate Gradient step
    Ap0 = M.*afun(M.*p0);
    alpha(j) = (r0(:)'*r0(:)) / (p0(:)'*Ap0(:));
    x = x + alpha(j)*p0;
    r1 = r0 - alpha(j)*Ap0;
    beta(j) = (r1(:)'*r1(:)) / (r0(:)'*r0(:));
    p1 = r1 + beta(j)*p0;
    
    %Compute norm
    rnorm = norm(r1);
    resV(j+1) = rnorm;
    xs(:,j+1) = x(:);
    
    %Compute the Lanczos vectors : v = scalar * r
    V(:,j+1) = r1(:);
    
    %Next step
    p0 = p1;
    r0 = r1;
    j = j+1;
end
%Assign output
if rnorm/bnorm > tol
    %If MCG iterated maxIter without convergence, return the solution
    flag = 1;
    [~, jmin] = min(resV);
    x = xs(:,jmin);
    rnorm = resV(jmin);
    nIter = jmin-1;
else
    flag = 0;
    nIter = j-1;
end
relres = rnorm/norm(b);
%resV = resV(1:(min(j,maxIter+1)));
info.xs = xs(:,1:(min(j,maxIter+1)));
V = V(:,1:(min(j,maxIter+1)));
xMin = M.*reshape(x,256,256);
if nargin > 6
    for k = 1:min(j,maxIter+1)
        info.errA(k) = sqrt((xs(:,k)-xtrue)'*afun(xs(:,k)-xtrue));
    end
end
