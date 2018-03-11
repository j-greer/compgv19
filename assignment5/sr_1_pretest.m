% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];
                     
% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations
debug = 0; % Debugging parameter will switch on step by step visualisation of quadratic model and various step options

% Starting point
x0 = [10; 10]; 

% Trust region parameters 
eta = 0.1;  % Step acceptance relative progress threshold
Delta = 1; % Trust region radius

% Minimisation with 2d subspace and dogleg trust region methods
Fsr1 = rmfield(F,'d2f');
[xTR_SR1, fTR_SR1, nIterTR_SR1, infoTR_SR1] = trustRegion(Fsr1, x0, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter, debug);

xMin = [0; 0];