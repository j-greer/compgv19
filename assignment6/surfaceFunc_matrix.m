function [value, fullM] = surfaceFunc_matrix(interiorM, boundaryM, dim)
% SURFACEFUNC_MATRIX evaluates the function to minimise the surface given a boundary square
% Problems 7.7 & 7.8 from Nocedal-Wright
%function value = surfaceMatrix(interiorM, boundaryM, dim)
%
% INPUTS
% interiorM: matrix to evaluate the function, size (dim)x(dim)x nPoints
%   - nPoints is the number of points to evaluate
% boundaryM: matrix that specifies the boundary values, size (dim+2)x(dim+2)
% dim: dimension of the problem, q-1
%
% OUTPUTS
% value: evaluation of the function at the given points
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rul·lan

% Find the number of points to evaluate. Each matrix in the 3rd dimension represents a point where the function  is evaluated.
 nPoints = size(interiorM, 3);
% Build the complete matrix. Surround the given interiorM with zeros to sum it with the given boundaryM.
interiorM_dim = [zeros(1, dim+2, nPoints); ...
                 zeros(dim, 1, nPoints) interiorM zeros(dim, 1, nPoints); zeros(1, dim+2, nPoints)];
% Build a binary boundary matrix of the form
% 1 1 1 1
% 1 0 0 1
% 1 0 0 1
% 1 1 1 1
% to multiply it by the given boundaryM and consider only boundary points.
binaryBoundaryM = [ones(1, dim+2); ones(dim, 1) zeros(dim, dim) ones(dim, 1); ones(1, dim+2)];
% Create the full matrix by adding the interior matrix interiorM and the corresponding boundaryM.
% We use repmat to construct a matrix for each given point.
fullM = interiorM_dim + repmat(binaryBoundaryM.*boundaryM, [1 1 nPoints]);

% Build diagonal matrices 
D11 = fullM(1:end-1, 1:end-1, :); % Superior-left submatrix
D12 = fullM(1:end-1, 2:end, :);   % Superior-right submatrix
D21 = fullM(2:end, 1:end-1, :);   % Inferior-left submatrix
D22 = fullM(2:end, 2:end, :);     % Inferior-right submatrix

% Compute the value of the element functions given in exercise 7.7 with the provided formula.
Dtotal = 1/(dim+1)^2*sqrt(1+(dim+1)^2*(D11-D22).^2 + (D12-D21).^2/2);
% Sum all the element functions to compute the function to minimise.
value = sum(sum(Dtotal, 1), 2);
% Permute value to give the function values along the 2nd dimension (columns)
value = permute(value, [1 3 2]);

