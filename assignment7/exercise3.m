% Numerical Optimisation: assignment 7
% Keshav Iyengar

clear variables;
close all;

%phiFun = @(x,t) (x(1) + x(2)*t^2)*exp(-x(3)*t);
t = 0:4/200:4;
phi = @(x) getPhi(x,t);
max_phi = 20.7146; %max value found
sigma = 0.05*max_phi;
x0 = [3; 150;2];
measurements = phi(x0) + mvnrnd(0,sigma,size(t,2),1);
figure;
plot(t,measurements);hold on;


F.f = @(x) 0.5*sum(getPhi(x,t) - measurements);
F.J = @(x) getJ(x,t);
F.r = @(x) getPhi(x,t) - measurements;
F.df = @(x) (getJ(x,t))'*(getPhi(x,t) - measurements);

% Starting point
x0 = [3;150;2];

% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations

% Line search parameters
alpha0 = 1;

% Strong Wolfe LS
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.5; % 0.1 Good for Newton, 0.9 - good for steepest descent, 0.5 compromise.
lsFun = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);

[xMin, fMin, nIter, info] = descentLineSearch(F, 'gauss', lsFun, alpha0, x0, tol, maxIter);
gauss_newton_results = phi(xMin);
plot(t,gauss_newton_results);

% Trust region parameters
eta = 0.1;  % Step acceptance relative progress threshold
Delta = 1; % Trust region radius
debug = 0;
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations
x0 = [3;150;2];

[x_k, f_k, k, info] = trustRegionLS(F, x0, @(F,x_k,Delta) solverCMlevenberg(F,x_k,Delta,maxIter), Delta, eta, tol, maxIter, debug);
