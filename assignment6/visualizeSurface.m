function visualizeSurface(info, X, Y, boundaryM, mode)
% VISUALIZESURFACES Convergence plot of iterates
%function visualizeConvergence(info,X,Y,Z,mode)
% INPUTS
% info: structure containing iteration history
%   - xs: taken steps
%   - xind: iterations at which steps were taken
%   - stopCond: shows if stopping criterium was satisfied, otherwsise k = maxIter
%   - Deltas: trust region radii
%   - rhos: relative progress
% X,Y: grid as returned by meshgrid
% boundaryM: matrix containing the boundary values
% mode: choose from {'final', 'iterative'}
%   'final': plot all iterates at once
%   'iterative': plot the iterates one by on to see the order in which steps are taken
% 
% Copyright (C) 2017 Marta M. Betcke, Kiko Rullan 

% Reshape iterations. The points of the iteration are given along the 2nd dimension. 
xs = permute(info.xs, [1 3 2]);
% Compute the dimension of the problem
dim = length(X)-2;

figure;
switch mode
  case 'final'  
    % Plot final surface
    xsM = reshape(xs(:, end), [dim, dim]);
    [value, Z] = surfaceFunc_matrix(xsM, boundaryM, dim);
    contourf(X, Y, Z, 20); 
    title('Convergence');
    
  case 'iterative'    
    % Plot the iterates one by one to see the order in which steps are taken
    nIter = size(info.xs,2);
    for j = 1:nIter
      xsM = reshape(xs(:, j), [dim, dim]);
      [value, Z] = surfaceFunc_matrix(xsM, boundaryM, dim);           
      contourf(X, Y, Z, 30); % Evaluate each contour
      title(['Convergence: steps 1 : ' num2str(j)])
      pause(0.5);
    end
         
end
