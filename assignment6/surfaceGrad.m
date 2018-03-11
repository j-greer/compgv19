function grad = surfaceGrad(x, boundaryM, dim, delta)
% GRADIENTSURFACE computes the gradient of the function to minimise the surface given a boundary square
% Problems 7.7 & 7.8 from Nocedal-Wright
%function grad = gradientSurface(x, boundaryM, dim)
%
% INPUTS
% x: column vector that provides the values for the interior matrix size (dim)*(dim)x1
% boundaryM: matrix that specifies the boundary values, size (dim+2)x(dim+2)
% dim: dimension of the problem, q+1
% delta: increment to compute the gradient
%
% OUTPUTS
% grad: gradient of the function
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rul·lan

% Compute the increment vector to evaluate the gradient.
% Recall that dim*dim is the number of variables of the function to minimise.
% incrementV is a matrix of the form
% delta 0   0    0
% 0  delta  0    0
% 0    0  delta  0
% 0    0    0   delta
incrementV = diag(delta*ones(dim*dim, 1));

% Positive vector of the central differences.
% repmat is used to repeat the given point along dim*dim variables 
xPos = repmat(x, [1 dim*dim]) + incrementV;
% The vector xPos has the following form with dim x dim variables
%        / x1+delta x1      x1     x1     \
% xPos = |   x2   x2+delta  x2     x2     |
%        |   x3     x3   x3+delta  x3     |
%        \   x4     x4      x4   x4+delta /

% Negative vector of the central differences
xNeg = repmat(x, [1 dim*dim]) - incrementV;
% The vector xNeg has the following form with dim x dim variables
%        / x1-delta x1      x1     x1     \
% xNeg = |   x2   x2-delta  x2     x2     |
%        |   x3     x3   x3-delta  x3     |
%        \   x4     x4      x4   x4-delta /

% Evaluate xPos and xNeg
surfPos = surfaceFunc_vector(xPos, boundaryM, dim);
surfNeg = surfaceFunc_vector(xNeg, boundaryM, dim);
% Compute the gradient using central differences.
% Each column contains the derivative with respect to a variable.
grad = (surfPos - surfNeg)/(2*delta);
% Permute grad to give the gradient along the 1st dimension (column)
grad = permute(grad, [2 1]);


