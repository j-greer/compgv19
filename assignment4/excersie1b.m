%% Initialisation
n = 1e2;
tol = 1e-12;
maxIter = 1e3;
% Initial point
x0 = zeros(n, 1);

% Matrix definition - Use sparse matrices
b = ones(n, 1);
A1 = spdiags((1:n)', 0, n, n); 

% Define xtrue
xtrue = zeros(n,1); 
xtrue(floor(n/4):floor(n/3)) = 1;
xtrue(floor(n/3)+1:floor(n/2)) = -2;
xtrue(floor(n/2)+1:floor(3/4*n)) = 1/2;

% Alternative xtrue
%  [V, Lambda] = eig(full(A3));
%  k = 5;
%  xtrue = V(:,1:5)*randn(5,1);
  

%% SOLVE - Without preconditioning

% Identity operator
M = @(y) y; 

% Linear preconditioned Conjugate gradient backtracking line search
[xMin1, nIter1, resV1, infoCG1] = conjugateGradient(A1, A1*xtrue, tol, maxIter, M, x0, xtrue);

% Run learner solution.
norm1 = norm(xMin1-xtrue);

%% SOLVE - Following A matrices and compare Convergence
A1 = diag(1:n);
A2 = diag([ones(n-1,1);100]);
%1d negative Laplace
A3 = -diag(ones(n-1, 1), -1) - diag(ones(n-1, 1), 1) + diag(2*ones(n, 1));

% Linear preconditioned Conjugate gradient backtracking line search
[xMin1, nIter1, resV1, infoCG1] = conjugateGradient(A1, A1*xtrue, tol, maxIter, M, x0, xtrue);
%convergence rate
eig_A1 = eig(A1);
k_A1 = eig_A1(end)/eig_A1(1);
rate1 = [];
for k=1:5
    rate1 = [rate1 2*((sqrt(k_A1) - 1)/(sqrt(k_A1) + 1))^k * norm(x0 - xtrue)];
end
[xMin2, nIter2, resV2, infoCG2] = conjugateGradient(A2, A2*xtrue, tol, maxIter, M, x0, xtrue);
%convergence rate
eig_A2 = eig(A1);
k_A2 = eig_A2(end)/eig_A2(1);
rate2 = [];
for k=1:5
    rate2 = [rate2 2*((sqrt(k_A2) - 1)/(sqrt(k_A2) + 1))^k * norm(x0 - xtrue)];
end

[xMin3, nIter3, resV3, infoCG3] = conjugateGradient(A3, A3*xtrue, tol, maxIter, M, x0, xtrue);
norm3 = norm(xMin3-xtrue);
%convergence rate
eig_A3 = eig(A3);
k_A3 = eig_A3(end)/eig_A3(1);
rate3 = [];
for k=1:5
    rate3 = [rate3 2*((sqrt(k_A3) - 1)/(sqrt(k_A3) + 1))^k * norm(x0 - xtrue)];
end

%plotting

figure;
plot(vecnorm(infoCG1.xs - xtrue))
hold on;
plot(rate1)
legend('real convergence','theoretical convergence')

figure;
plot(vecnorm(infoCG2.xs - xtrue))
hold on;
plot(rate2)
legend('real convergence','theoretical convergence')

figure;
plot(vecnorm(infoCG2.xs - xtrue))
hold on;
plot(rate2)
legend('real convergence','theoretical convergence')

