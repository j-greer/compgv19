% For computation define as function of 1 vector variable
F.f = @(x) (x(1) - 3*x(2)).^2 + x(1).^4;
F.df = @(x) [2*(x(1) - 3*x(2)) + 4*x(1).^3; -6*(x(1) - 3*x(2))];
F.d2f = @(x) [2 + 12*x(1).^2, -6; -6, 18];

% For visualisation proposes define as function of 2 variables
func = @(x,y) (x - 3*y).^2 + x.^4;
% Starting point
x0 = [10; 10]; 

% Parameters
maxIter = 200; 
tol = 1e-10; % Stopping tolerance on relative step length between iterations

%% BFGS Method
% Line search parameters
alpha0 = 1;

% Strong Wolfe LS
lsOpts_LS.c1 = 1e-4;
lsOpts_LS.c2 = 0.5; % 0.1 Good for Newton, 0.9 - good for steepest descent, 0.5 compromise.
lsFunS = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha0, lsOpts_LS);
lsFun = lsFunS;

% Minimisation with Newton, Steepest descent and BFGS line search methods
[xLS_BFGS, fLS_BFGS, nIterLS_BFGS, infoLS_BFGS] = descentLineSearch(F, 'bfgs', lsFun, alpha0, x0, tol, maxIter);

%% SR-1 Method
% Parameters
debug = 0; % Debugging parameter will switch on step by step visualisation of quadratic model and various step options

% Starting point
x0 = [10; 10]; 

% Trust region parameters 
eta = 0.1;  % Step acceptance relative progress threshold
Delta = 1; % Trust region radius

% Minimisation with 2d subspace and dogleg trust region methods
Fsr1 = rmfield(F,'d2f');
[xTR_SR1, fTR_SR1, nIterTR_SR1, infoTR_SR1] = trustRegion(Fsr1, x0, @solverCM2dSubspaceExt, Delta, eta, tol, maxIter, debug);

n = 300;
x = linspace(-5,10,n+1);
y = linspace(-5,10,n+1);
[X,Y] = meshgrid(x,y);

figure;
hold on;
plot(infoLS_BFGS.xs(1, :), infoLS_BFGS.xs(2, :), '-or', 'LineWidth', ...
2, 'MarkerSize', 3); % only first 10 iterations
plot(infoTR_SR1.xs(1, :), infoTR_SR1.xs(2, :), '-*g', 'LineWidth', ...
2, 'MarkerSize', 3); % only first 10 iterations
contour(X, Y, log(max(func(X,Y), 1e-3)), 20);

figure;
semilogy(infoLS_BFGS.alphas(1:end), '-or', 'LineWidth', 2, 'MarkerSize', ...
2); hold on;
semilogy(infoTR_SR1.rhos(1:end), '-*g', 'LineWidth', ...
2, 'MarkerSize', 2);


%% Exercise 3 part b
diffBFGS = zeros(1,size(infoLS_BFGS.xs,2)-1);
for i=1:size(infoLS_BFGS.xs,2)-1
   H_BFGS = infoLS_BFGS.H{i};
   diffBFGS(i) = norm(eye(2) - H_BFGS(eye(2))/(F.d2f(infoLS_BFGS.xs(:,i))));
end

diffSR1 = zeros(1,size(infoTR_SR1.xs,2)-1);
for i=1:size(infoTR_SR1.xs,2)-1
   B_SR1 = infoTR_SR1.B{i};
   diffSR1(i) = norm(B_SR1 - (F.d2f(infoTR_SR1.xs(:,i))));
end
figure
semilogy(diffBFGS)
hold on
semilogy(diffSR1)
legend('BFGS','SR1')
