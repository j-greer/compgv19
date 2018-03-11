function [x, flag, relres, iter, resvec, xs, V] = mcg(A, b, tol, maxit, M, x0)
% MPCG Implicitely symmetric preconditioned CG
%   Notice that the left preconditioned NE (with M inner product) and
%   the right preconditioned NE (with M^-1 inner product) produce the same algorithm, 
%   hence only one version.

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

n = length(b);

if nargin < 6
    x0 = zeros(n,1);
end

%Initialize:
x = x0;            % Intial x
r0 = b - afun(x);  % residual: b - Ax
z0 = mfun(r0);     % M^(-1)r0
p0 = z0;           % conjugate gradient
rnorm = norm(r0);  % residual norm |r|
bnorm = rnorm;

alpha = zeros(maxit,1);
beta = zeros(maxit,1);
resvec = zeros(maxit+1,1);
resvec(1) = rnorm; 
xs = zeros(n, maxit+1);
V = zeros(n, maxit+1);
xs(:,1) = x;
V(:,1) = z0;
j = 1;
while j <= maxit && rnorm/bnorm >= tol %|r|/|b| > tol

    %Preconditioned Conjugate Gradient step
    Ap0 = afun(p0);
    alpha(j) = (r0'*z0) / (Ap0'*p0); 
    x = x + alpha(j)*p0; 
    r1 = r0 - alpha(j)*Ap0;
    z1 = mfun(r1);
    beta(j) = (r1'*z1) / (r0'*z0);
    p1 = z1 + beta(j)*p0;
    
    %Compute norm
    rnorm = norm(r1);
    resvec(j+1) = rnorm;
    xs(:,j+1) = x;
    %Compute the Lanczos vectors : v = scalar * r
    V(:,j+1) = z1;
    
    %Next step
    p0 = p1;
    r0 = r1;
    z0 = z1;    
    j = j+1;
end

%Assign output
if rnorm/bnorm > tol,
  %If MCG iterated maxit without convergence, return the solution with the minimal residual
    flag = 1;
    [~, jmin] = min(resvec);
    x = xs(:,jmin);
    rnorm = resvec(jmin);
    iter = jmin-1;
else 
    flag = 0;
    iter = j-1;
end

relres = rnorm/norm(b);
resvec = resvec(1:(min(j,maxit+1)));
xs = xs(:,1:(min(j,maxit+1)));
V = V(:,1:(min(j,maxit+1)));
