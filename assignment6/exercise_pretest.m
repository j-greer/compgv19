
% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];

% Starting point
x0 = [5; -7]; 

% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations

% Line search parameters
alpha0 = 1;

% Strong Wolfe LS
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.5; % 0.1 Good for Newton, 0.9 - good for steepest descent, 0.5 compromise.
lsFunS = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
lsFun = lsFunS;

% Minimisation with Newton, Steepest descent and BFGS line search methods
[xLS_NCG, fLS_NCG, nIterLS_NCG, infoLS_NCG] = descentLineSearch(F, 'newton-cg', lsFun, alpha0, x0, tol, maxIter)