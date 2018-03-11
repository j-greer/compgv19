% For computation define as function of 1 vector variable
F.f = @(x) 100.*(x(2) - x(1)^2).^2 + (1 - x(1)).^2; % function handler, 2-dim vector
F.df = @(x) [-400*(x(2) - x(1)^2)*x(1) - 2*(1 - x(1)); 
              200*(x(2) - x(1)^2)];  % gradient handler, 2-dim vector
F.d2f = @(x) [-400*(x(2) - 3*x(1)^2) + 2, -400*x(1); -400*x(1), 200]; % hessian handler, 2-dim vector
% For visualisation proposes define as function of 2 variables
rosenbrock = @(x,y) 100.*(y - x.^2).^2 + (1 - x).^2;
% Initialisation
alpha0 = 1;
maxIter = 300;
alpha_max = alpha0;
tol = 1e-6;
%=============================
% Point x0 = [1.2; 1.2]
%=============================
x0 = [1.2;1.2];

%%
%Apply Steepest Descent (with backtracking line search)
btopts.rho = 0.1;
btopts.c1 = 1e-4;

lsFun = @(x_k, p_k, alpha0) backtracking(F, x_k, p_k, alpha_max, btopts);
[xSteep, fSteep, nIterSteep, infoSteep] = descentLineSearch(F, 'steepest', lsFun, alpha0, x0, tol, maxIter);

%%
%Apply Newton's Method (with backtracking line search)
btopts.rho = 0.1;
btopts.c1 = 1e-4;

lsFun = @(x_k, p_k, alpha0) backtracking(F, x_k, p_k, alpha_max, btopts);
[xNewt, fNewt, nIterNewt, infoNewt] = descentLineSearch(F, 'newton', lsFun, alpha0, x0, tol, maxIter);


%%
% Define grid for visualisation
n = 300;
x = linspace(0.7,1.5,n+1);
y = x;
[X,Y] = meshgrid(x,y);


figure;
hold on;
plot(infoSteep.xs(1, :), infoSteep.xs(2, :), '-or', 'LineWidth', ...
2, 'MarkerSize', 3); % only first 10 iterations
plot(infoNewt.xs(1, :), infoNewt.xs(2, :), '-*g', 'LineWidth', ...
2, 'MarkerSize', 3); % only first 10 iterations
contour(X, Y, log(max(rosenbrock(X,Y), 1e-3)), 20);
title('x_k: descent with line search')
legend('Steepest descent', 'Newton');

% Step length plot
figure;
semilogy(infoSteep.alphas(1:end), '-or', 'LineWidth', 2, 'MarkerSize', ...
2); hold on;
semilogy(infoNewt.alphas(1:end), '-*g', 'LineWidth', ...
2, 'MarkerSize', 2);
grid on;
title('alpha_k: backtracking');
legend('Steepest Descent', 'Newton');