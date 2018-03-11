function value = surfaceFunc_vector(x, boundaryM, dim)
% SURFACEFUNC_VECTOR evaluates the function to minimise the surface given a boundary square
% Problems 7.7 & 7.8 from Nocedal-Wright
%function value = surfaceVector(x, boundaryM, dim)
%
% INPUTS
% x: matrix that provides the values for the interior matrix size (dim)*(dim)x nPoints
%   - the number of columns is the number of points to evaluate
% boundaryM: matrix that specifies the boundary values, size (dim+2)x(dim+2)
% dim: dimension of the problem, q+1
%
% OUTPUTS
% value: evaluation of the function
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rul·lan

% Find the number of points to evaluate. Each column represents a point where the function is evaluated.
nPoints = size(x, 2);
% Reshape the vector.
% The vector is reshaped according to the input expected from surfaceMatrix, which is a stacking matrices along the 3rd dimension to evaluate points.
interiorM = reshape(x, [dim, dim, nPoints]);
% Evaluate the function at the given points using surfaceMatrix
value = surfaceFunc_matrix(interiorM, boundaryM, dim);
