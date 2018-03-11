function p = solverCM2dSubspaceExtLS(F, x_k, Delta)
% SOLVERCM2DSUBSPACEEXT Solves quadratic constraint trust region problem with possibly indefinite Hessian via 2d subspace
% p = solverCM2dSubspaceExt(F, x_k, Delta)
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
% x_k: current iterate
% Delta: trust region radius
% OUTPUT
% p: step (direction times lenght)
%
% Copyright (C) 2017 Marta M. Betcke, Kiko Rullan 

% Parameters for mcg
tol = 1e-6;
maxit = 1000;

% Compute gradient and Hessian (as a handle)
g = F.df(x_k); 
if strcmp(class(F.d2f), 'function_handle')
  if nargin(F.d2f) > 1
    B = @(q) F.d2f(x_k, q); % Hessian approximation is appropriate form
  else
    B = @(q) F.d2f(x_k)*q;  % Hessian at x as a function of x 
  end
else
  B = @(q) F.d2f*q;         % Hessian as a matrix 
end

% Compute smallest eigenvalue of the approximate Hessian using iterative solver in eigs (arpack). This only works if 
% the dimension is large enough, so has to be saveguarded with eig for small case.
if length(x_k) > 10
  opts.tol = tol; opts.maxit = maxit; %opts.issym = 1; 
  % opts.disp = 1; % ARPACK info 
  %lambdasB = eigs(B, length(x_k), min(length(x_k),6),'smallestreal', opts);
  lambdasB = eigs(B, length(x_k), 6, 'SR', opts);
else
  % Quick & dirty test
  lambdasB = eig(B(eye(length(x_k))));
end
lambdaB1 = min(lambdasB); 

% Three cases: B has
% (1) negative eigenvalues
% (2) zero eigenvalues
% (3) positive eigenvalues
% Handle indefinite Hessian first cases (1), (2).
if abs(lambdaB1) < 100*tol^2 % (2) zero eigenvalue(s) 
  % Take Cauchy point step 
  gTBg = g'*B(g);
  if gTBg <= 0
    tau = 1;   
  else
    tau = min(norm(g)^3/(Delta*gTBg), 1);
  end
  p = -tau*Delta/norm(g)*g;
  return;
  
elseif lambdaB1 < 0 % (1) negative eigenvalue(s)
  alpha = -1.5*lambdaB1; % shift ensuring that B + alpha*I is p.d.
  
  B = @(q) B(q) + alpha*q; % modify to spd matrix (B + alpha I)
  % unconstraint solution with modified Hessian
  [pNewt, flag, relres, iter, resvec, pNewts] = mcg(B, g, tol, maxit, @(x) x); % iterative solve to approximate B\g;
  
  if norm(pNewt) <= Delta 
    npNewt = pNewt/norm(pNewt);
    v = randn(size(x_k));
    v = v/norm(v);
    v = -0.1*npNewt + 0.1*(v - npNewt*npNewt'*v); % v: v'*pNewt <= 0    
    p = -pNewt + v;  % ensure ||p|| >= ||pNewt||
    %p = -1.1*pNewt;  % ensure ||p|| >= ||pNewt||
    %p = -npNewt*Delta;  % ensure ||p|| >= ||pNewt||
    return;
    
  else % unconstraint solution not in the trust region, construct 2d subspace and proceed as in p.d. case.
    % Orthonormalize the 2D projection subspace
    V = orth([g, pNewt]);
  end
  
else % (3) positive eigenvalues
  eps_k = min(0.5, sqrt(norm(g)))*norm(g); % norm(g) factor is to compensate for relative residual in MCG
  [pNewt, flag, relres, iter, resvec, pNewts] = mcg(B, g, eps_k, maxit, @(x) x); % iterative solve to approximate B\g;
  % Orthonormalize the 2D projection subspace
  V = orth([g, pNewt]);
end

% Check if gradient and Newton steps are collinear. If so return Cauchy point.
if size(V,2) == 1
  % Calculate Cauchy point
  gTBg = g'*B(g);
  if gTBg <= 0
    tau = 1;   
  else
    tau = min(norm(g)^3/(Delta*gTBg), 1);
  end
  p = -tau*Delta/norm(g)*g;
  return;
end

% To constraint the optimisation to subspace span([g, B\g]),  
% we express the solution i.e. the direction a linear combination 
% p = V*a with 'a' being a vector of two coefficients. 
% Substituting p = V*a into the quadratic model
%
%     m(p) = f(x_k) + g'*p + 0.5*p'*B*p with g = df(x_k), B = d2f(x_k) 
%     s.t. p'*p <= Delta^2 
%
% we obtain the projected model, which is a quadratic model for 'a'
%
%     mv(a) = f(x_k) + gv'*a + 0.5*a'*Bv*a 
%     s.t. a'*a <= Delta^2      (due to V'*V = I)
%
% Furthermore, as long as V has a full rank (g and B\g 
% are not collinear), if B is s.p.d. so is Bv. 
% Note, that if g = c*B\g the problem becomes 1D.  

% Project on V
%Bv = V'*B(V);
Bv = V'*[B(V(:, 1)) B(V(:, 2))]; % efficient implementation of B is only valid for single vector
gv = V'*g;

% To solve the projected model mv subject to p'*p <= Delta^2
% we make use of Theorem 4.1 Nocedal Wiright.
% From this theorem for mv we have that 'a' 
% minimizes mv s.t. a'*a <= Delta^2 iff
%
%     (Bv + lambda*I) * a = -gv,    lambda >= 0
%     lambda * (Delta^2 - a'*a) = 0
%     (Bv + lambda * I) is s.p.d.
%
% This gives two cases:
% (1) lambda = 0  &  a'*a < Delta^2 (the unconstraint solution 
%     is inside the trust region).
%     Then the first equation becomes Bv * a = -gv i.e. a = -Bv\gv; 
% (2) lambda>= 0  &  a'*a = Delta^2 (the constraint is active)
%     Then we can solve the first equation 
%     (E1)    a = -(Bv + lambda*I) \ gv
%     The additional equation is provided by the constraint 
%     (E2)    a'*a = Delta^2
%     To solve this system we make use or eigendecomposition 
%     of Bv = Q*Lambdas*Q' with Q orthonormal
%         Q'*a = - inv(Lambdas + lambda*I) * Q'*gv
%     and realise that (Q'*a)'*(Q'*a) = a'*Q*Q'*a = a'*a.
%     We denote Qa = Q'*a and Qg = Q'*gv. 
%     For ith element on Qa, 
%         Qa(i) = - 1/(lambdas(i) + lambda) * Qg(i), 
%     with lambdas(i) = Lambdas(i,i). 
%     Substituting Qa into Qa'*Qa = Qa(1)^2 + Qa(2)^2 = Delta^2 we obtain 
%         Qg(1)^2/(lambdas(1) + lambda)^2 + Qg(2)^2/(lambdas(2) + lambda)^2 = Delta^2
%     which we transform to 4th degree polynomial in lambda 
%     (assuming that lambdas(i) + lambda > 0)
%         r(1) lambda^4 + r(2) lambda^3 + r(3) lambda^2 + r(4) lambda + r(5) = 0 


% Case (1)
% Compute unconstrained solution and check if it lies in the trust region
a = -Bv\gv;
if a'*a < Delta^2
  % Compute the solution p
  p = V*a;
  return;
end

% Case (2)
[Q, Lambdas] = eig(Bv);
lambdas = diag(Lambdas);
Qg = Q'*gv;

r(5) = Delta^2*lambdas(1)^2*lambdas(2)^2 - Qg(1)^2*lambdas(2)^2 - Qg(2)^2*lambdas(1)^2;

r(4) = 2*Delta^2*lambdas(1)^2*lambdas(2) + 2*Delta^2*lambdas(1)*lambdas(2)^2 ...
      -2*Qg(1)^2*lambdas(2) - 2*Qg(2)^2*lambdas(1);

r(3) = Delta^2*lambdas(1)^2 + 4*Delta^2*lambdas(1)*lambdas(2) + Delta^2*lambdas(2)^2 ...
      -Qg(1)^2- Qg(2)^2; 
    
r(2) = 2*Delta^2*lambdas(1) + 2*Delta^2*lambdas(2);

r(1) = Delta^2;

% Compute roots of the polynomial and select positive one
rootsR = roots(r);
%rootsR = rootsR(rootsR >= 0);
lambda = min(rootsR(rootsR + min(lambdas) > 0));
%lambda = min(rootsR);


% Compute a from   Qa(i) = - 1/(lambdas(i) + lambda) * Qg(i)
a = Q* ( (-1./(lambdas(:) + lambda)) .* Qg);
% Compute the solution p
p = V*a;
% Renormalize to ||p|| = Delta, because the condition number of the polynomial root finder is high
p = Delta/norm(p)*p;

