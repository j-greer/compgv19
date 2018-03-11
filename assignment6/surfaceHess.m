function hess = surfaceHess(x, boundaryM, dim, delta)
% HESSIANSURFACE computes the hessian of the function to minimise the surface given a boundary square
% Problems 7.7 & 7.8 from Nocedal-Wright
%function hess = hessianSurface(x, boundaryM, dim, delta)
%
% INPUTS
% x: column vector that provides the values for the interior matrix size (dim)*(dim)x1
% boundaryM: matrix that specifies the boundary values, size (dim+2)x(dim+2)
% dim: dimension of the problem, q+1
% delta: increment to compute the hessian
%
% OUTPUTS
% hess: hessian of the function
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rul·lan

% Compute the increment vector and gradients to compute the hessian
% Recall that (dim)x(dim) is the number of variables of the function to minimise.
% incrementV is a matrix of dimension (dim*dim)x(dim*dim) of the form
% delta 0   0    0
% 0  delta  0    0
% 0    0  delta  0
% 0    0    0   delta
incrementV = diag(delta*ones(dim*dim, 1));


% Initialise the hessian matrix with dim*dim x dim*dim zeros
hess = zeros(dim*dim);

% Iterate over each variable to compute the hessian.
% The idea is to take central differences of the gradient for each variable. We know:
%              / df/dx1 \
%              | df/dx2 |
% grad(f(x)) = | df/dx3 |
%              \ df/dx4 /
% Taking central differences again for a variable in each column leads to:
%              / d2f/(dx1)^2   d2f/(dx1*dx2) d2f/(dx1*dx3) d2f/(dx1*dx4) \
%              | d2f/(dx2*dx1) d2f/(dx2^2)   d2f/(dx2*dx3) d2f/(dx2*dx4) |
% hess(f(x)) = | d2f/(dx3*dx1) d2f/(dx3*dx2) d2f/(dx3^3)   d2f/(dx3*dx4) |
%              \ d2f/(dx4*dx1) d2f/(dx4*dx2) d2f/(dx4*dx3) d2f/(dx4^2)   /
for n = 1:dim*dim
    % Positive gradient vector of the central differences
    gradPos = surfaceGrad(x + incrementV(:, n), boundaryM, dim, delta);
    % Negative gradient vector of the central differences
    gradNeg = surfaceGrad(x - incrementV(:, n), boundaryM, dim, delta);
    % Compute the n-th column of the hessian, corresponding to the derivative with respect to the n-th variable                                                                           
    hess(:, n) = (gradPos - gradNeg)/(2*delta);
end
