function [x, f, nIter, info] = ex2c(x0,type)

% For computation define as function of 1 vector variable
F.f = @(x) x(1)^2 + 5*x(1)^4 + 10*x(2)^2;
F.df = @(x) [2*x(1) + 20*x(1)^3; 20*x(2)];
F.d2f = @(x) [2 + 60*x(1)^2, 0; 0, 20];

% Initialisation
alpha0 = 1;
tol = 1e-8;
maxIter = 100;

lsOptsCG_LS.c1 = 1e-4;
lsOptsCG_LS.c2 = 0.1;
lsFun = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOptsCG_LS);
[x, f, nIter, info] = nonlinearConjugateGradient(F, lsFun, type, alpha0, x0, tol, maxIter);
end