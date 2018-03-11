function p = solverCM2dSubspace(F, x_k, Delta)
% SOLVERCM2DSUBSPACE Solves quadratic constraint trust region problem via 2d subspace
% p = solverCM2dSubspace(F, x_k, Delta)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% x_k: current iterate
% Delta: trust region radius
% OUTPUT
% p: step (direction times length)
% Copyright (C) 2017 Marta M. Betcke, Kiko Rullan
%compute gradient and hessian
g = F.df(x_k);
B = F.d2f(x_k);
%determine eigenvalues of B and smallest eigenvalue
lambdasB = eig(B);
lambdaB1 = min(lambdasB);

%Check for different cases
%1) negative eigenvalues
%2) zero eigenvalues

if lambdaB1 < eps %zero eigenvalues
    %do cauchy point
   gTBg = g'*B*g;
   if gTBg <= 0
      tau = 1;
   else
       tau = min(norm(g)^3/(Delta*gTBg),1);
   end
   p = -tau*Delta/norm(g)*g;
   return
   
elseif lambdaB1 < 0 %negative eigenvalues
    
end

