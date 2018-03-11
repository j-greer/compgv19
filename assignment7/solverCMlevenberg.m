function p = solverCMlevenberg(F, x_k, Delta, maxIter)
% SOLVERCMLEVENBERG Levenberg-Marguardt solver for constraint trustregion problem
%function p = solverCMlevenberg(F, x_k, Delta, maxIter)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
%   - J: handler for the jacobian of r
%   - r: residual function
% x_k: current iterate
% Delta: upper limit on trust region radius
% maxIter: maximum number of iterations
%
% OUTPUT
% p: step (direction times lenght)
%
% Based on Algorithm 4.3 in Nocedal Wright
% Copyright (C) 2017 Marta M. Betcke, Kiko Rul·lan 

% Initialise
lambda = 1;
nIter = 0;
% Compute the QR factorisation at x_k
J = F.J(x_k);
r = F.r(x_k);
[m, n] = size(J);
[Q_ini, R_ini] = qr(J); % Q: m x m orthogonal, R: m x n upper triangular
%maxEigenval = max(eig(R'*R)); % limit the value of lambda

p = 0;
while (nIter < maxIter && abs(norm(p)-Delta)/Delta > 0.05 && lambda > 0)
    % Update the Cholesky factorisation
    Q = Q_ini;
    R = R_ini;
    for i = 1:n
      % Construct i-th row of sqrt(lambda)*I
      row = zeros(1, n);
      row(i) = sqrt(lambda);
      % Insert i-th row of sqrt(lambda)*I at position m+i below R and update QR decomposition
      [Q, R] = qrinsert(Q, R, m+i, row, 'row');
    end
    % Solve (R'*R) p = (-J'*r) for L-M direction p
    p = R\(R'\(-J'*r)); 
    % Compute q (eigenvector, see description of Algorithm 4.3 Nocedal Wright)
    q = R'\p; 
    % Update lambda (the Lagrange multiplyer for the trust region problem
    % and the shift to make J'*J spd). Note that J'*J is at least positive semidefinite 
    % so any positive shift will make it spd.
    %lambda = max(0, lambda + (norm(p)./norm(q)).^2*(norm(p) - Delta)/Delta);
    lambda = max(0, lambda + (sum(p.^2)./sum(q.^2))*(norm(p) - Delta)/Delta);
    nIter = nIter+1;
end
