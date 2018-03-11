% For computation define as function of 1 vector variable
F.f = @(x) 100.*(x(2) - x(1)^2).^2 + (1 - x(1)).^2; % function handler, 2-dim vector
F.df = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1)); 
              200*(x(2) - x(1)^2)];  % gradient handler, 2-dim vector
F.d2f = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200]; % hessian handler, 2-dim vector
% For visualisation proposes define as function of 2 variables
rosenbrock = @(x,y) 100.*(y - x.^2).^2 + (1 - x).^2;
%% Backtracking
% Initialisation
alpha0 = 1;
maxIter = 300;
alpha_max = alpha0;
tol = 1e-6;
%=============================
% Point x0 = [-1.2; 1]
%=============================
x0 = [-1.2;1.0];
%%
%Apply Steepest Descent (with Algorithm 3.5 line search)
lsOptsSteep.rho = 0.1;
lsOptsSteep.c1 = 1e-4;
lsFun = @(x_k, p_k, alpha0) backtracking(F, x_k, p_k, alpha_max, lsOptsSteep);
[xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(F, 'steepest', lsFun, alpha0, x0, tol, maxIter);
%%
%Apply Newton's Method (with Algorithm 3.5 line search)
lsOptsSteep.rho = 0.1;
lsOptsSteep.c1 = 1e-4;

lsFun = @(x_k, p_k, alpha0) backtracking(F, x_k, p_k, alpha_max, lsOptsSteep);
[xNewt, fNewt, nIterNewt, infoNewt] = descentLineSearch(F, 'newton', lsFun, alpha0, x0, tol, maxIter);

%%
% Define grid for visualisation
n = 300;
x = linspace(-1.5,1.5,n+1);
y = linspace(-0.5,2.5,n+1);
[X,Y] = meshgrid(x,y);

figure;
hold on;
plot(infoSteep.xs(1, :), infoSteep.xs(2, :), '-or', 'LineWidth', ...
2, 'MarkerSize', 3);
plot(infoNewt.xs(1, :), infoNewt.xs(2, :), '-*g', 'LineWidth', ...
2, 'MarkerSize', 3);
legend('Steepest Descent', 'Newton','rosenbrock');
contour(X, Y, log(max(rosenbrock(X,Y), 1e-3)), 20);
title('x_k: Backtracking');
% Step length plot
figure;
semilogy(infoSteep.alphas(1:end), '-or', 'LineWidth', 2, 'MarkerSize', ...
2); hold on;
semilogy(infoNewt.alphas(1:end), '-*g', 'LineWidth', ...
2, 'MarkerSize', 2);
grid on;
title('alpha_k: Backtracking');
legend('Steepest Descent', 'Newton');



%% Line Search Strong WC
% Initialisation
alpha0 = 1;
maxIter = 300;
alpha_max = alpha0;
tol = 1e-6;
%=============================
% Point x0 = [-1.2; 1]
%=============================
x0 = [-1.2;1.0];
%%
%Apply Steepest Descent (with Algorithm 3.5 line search)
lsOptsSteep.c1 = 1e-4;
lsOptsSteep.c2 = 0.1;

lsFun = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha_max, lsOptsSteep);
[xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(F, 'steepest', lsFun, alpha0, x0, tol, maxIter);
%%
%Apply Newton's Method (with Algorithm 3.5 line search)
lsOptsSteep.c1 = 1e-4;
lsOptsSteep.c2 = 0.9;

lsFun = @(x_k, p_k, alpha0) lineSearch(F, x_k, p_k, alpha_max, lsOptsSteep);
[xNewt, fNewt, nIterNewt, infoNewt] = descentLineSearch(F, 'newton', lsFun, alpha0, x0, tol, maxIter);

%%
% Define grid for visualisation
n = 300;
x = linspace(-1.5,1.5,n+1);
y = linspace(-0.5,2.5,n+1);
[X,Y] = meshgrid(x,y);

figure;
hold on;
plot(infoSteep.xs(1, :), infoSteep.xs(2, :), '-or', 'LineWidth', ...
2, 'MarkerSize', 3);
plot(infoNewt.xs(1, :), infoNewt.xs(2, :), '-*g', 'LineWidth', ...
2, 'MarkerSize', 3);
legend('Steepest Descent', 'Newton','rosenbrock');
contour(X, Y, log(max(rosenbrock(X,Y), 1e-3)), 20);
title('x_k: line search strong WC');
% Step length plot
figure;
semilogy(infoSteep.alphas(1:end), '-or', 'LineWidth', 2, 'MarkerSize', ...
2); hold on;
semilogy(infoNewt.alphas(1:end), '-*g', 'LineWidth', ...
2, 'MarkerSize', 2);
grid on;
title('alpha_k: line search strong WC');
legend('Steepest Descent', 'Newton');
