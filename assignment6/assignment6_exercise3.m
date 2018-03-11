%% *COMPGV19: Assignment 6*
%
% Marta Betcke and Kiko Rul·lan
%

%%
% *Exercise 3*
% Minimise the surface given by the boundary conditions using Algorithm 7.1

close all;
clear variables;

newton_ls = true;
bfgs = true;
sr1 = true;

%==============================
% Initialisation
%==============================
% Dimensions
q = 14; % dimension of interior square
n = q+2; % dimension of square with boundary
% Boundary
s = linspace(0,1-1/n,n-1);
leftBoundary   = sin(pi*s+pi);   % left boundary
rightBoundary  = sin(3*pi*s+pi); % right boundary
topBoundary    = sin(3*pi*s);    % top boundary
bottomBoundary = sin(pi*s);      % bottom boundary
bndFun = [leftBoundary topBoundary rightBoundary bottomBoundary];
figure, plot(bndFun);
% Indicies of the boundary nodes in counter clockwise order
bndInd = [1:q+2, (1:q)*(q+2) + ones(1,q)*(q+2), (q+1)*(q+2) + (q+2:-1:1), (q:-1:1)*(q+2) + ones(1,q)];
boundaryM = zeros(q+2,q+2);
boundaryM(bndInd(:)) = bndFun;
% Initialise surface to zero
x0 = zeros(q*q, 1);

%==============================
% Assign function handlers
%==============================
delta = 1e-2; % for computing numerical gradient
F.f = @(x) surfaceFunc_vector(x, boundaryM, q);   % Function handler
F.df = @(x) surfaceGrad(x, boundaryM, q, delta);  % Gradient handler
F.d2f = @(x) surfaceHess(x, boundaryM, q, delta); % Hessian handler

%==============================
% LS-Newton-CG
%==============================
% YOUR CODE HERE
% Parameters
maxIter = 200;
tol = 1e-5; % Stopping tolerance on relative step length between iterations

% Line search parameters
alpha0 = 1;

% Strong Wolfe LS
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.5; % 0.1 Good for Newton, 0.9 - good for steepest descent, 0.5 compromise.
lsFunS = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
lsFun = lsFunS;

if newton_ls
    tic
    % Minimisation with Newton, Steepest descent and BFGS line search methods
    [xLS_NCG, fLS_NCG, nIterLS_NCG, infoLS_NCG] = descentLineSearch(F, 'newton-cg',...
        lsFun, alpha0, x0, tol, maxIter);
    newton_time = toc;
end

%==============================
% BFGS
%==============================
% YOUR CODE HERE
if bfgs
    tic
    % Minimisation with Newton, Steepest descent and BFGS line search methods
    [xLS_BFGS, fLS_BFGS, nIterLS_BFGS, infoLS_BFGS] = descentLineSearch(F, 'bfgs',...
        lsFun, alpha0, x0, tol, maxIter);
    bfgs_time = toc;
end

%==============================
% TR-SR1
%==============================
% YOUR CODE HERE
debug = 0; % Debugging parameter will switch on step by step visualisation of
%quadratic model and various step options

% Trust region parameters
eta = 0.1;  % Step acceptance relative progress threshold
Delta = 1; % Trust region radius
if sr1
    tic
    Fsr1 = rmfield(F,'d2f');
    % Minimisation with 2d subspace and dogleg trust region methods
    [xTR_SR1, fTR_SR1, nIterTR_SR1, infoTR_SR1] = trustRegionLS(Fsr1, x0, ...
        @solverCM2dSubspaceExtLS, Delta, eta, tol, maxIter, debug);
    sr1_time = toc;
end

%==============================
% Visualize convergence
%==============================
X = 0:1/(q+1):1;
Y = 0:1/(q+1):1;

% SUBSTITUTE INFO_EXAMPLE by the corresponding info structure from LS-Newton-CG, BFGS, TR-SR1 obtained in the minimisation.
if newton_ls
info_example.xs = infoLS_NCG.xs;
visualizeSurface(info_example, X, Y, boundaryM, 'final');
disp(['Newton-LS time taken: ',newton_time])
end
if bfgs
info_example.xs = infoLS_BFGS.xs;
visualizeSurface(info_example, X, Y, boundaryM, 'final');
disp(['BFGS time taken: ',bfgs_time])
end
if sr1
info_example.xs = infoTR_SR1.xs;
visualizeSurface(info_example, X, Y, boundaryM, 'final');
disp(['SR1 time taken: ',sr1_time])
end



