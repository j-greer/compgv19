function [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter)
% DESCENTLINESEARCH Wrapper function executing  descent with line search
% [xMin, fMin, nIter, info] = descentLineSearch(F, descent, ls, alpha0, x0, tol, maxIter) 
%
% INPUTS
% F: structure with fields
%   - f: function handler
%   - df: gradient handler
%   - d2f: Hessian handler
%   - J: Jacobian handler (Gauss-Newton method)
%   - r: residual handler (Gauss-Newton method)
% descent: specifies descent direction {'steepest', 'newton', 'newton-cg', 'newton-reg', 'gauss'}
% alpha0: initial step length 
% x0: initial iterate
% tol: stopping condition on relative error norm tolerance 
%      norm(x_Prev - x_k)/norm(x_k) < tol;
% maxIter: maximum number of iterations
%
% OUTPUTS
% xMin, fMin: minimum and value of f at the minimum
% nIter: number of iterations 
% info: structure with information about the iteration 
%   - xs: iterate history 
%   - alphas: step lengths history 
%
% Copyright (C) 2017  Marta M. Betcke, Kiko Rullan

% Initialization
nIter = 0;
normError = 1;
x_k = x0;
info.xs = x0;
info.alphas = alpha0;

% Loop until convergence or maximum number of iterations
while (normError >= tol && nIter <= maxIter)
    
  % Increment iterations
    nIter = nIter + 1;

    % Compute descent direction
    switch lower(descent)
      case 'steepest'
        p_k = -F.df(x_k); % steepest descent direction
      case 'newton'
        p_k = -F.d2f(x_k)\F.df(x_k); % Newton direction
      case 'newton-cg'
        % Conjugate gradient method
        df_k = F.df(x_k); % gradient
        B_k = F.d2f(x_k); % hessian
        eps_k = min(0.5, sqrt(norm(df_k)))*norm(df_k);
        z_j = 0*df_k;
        r_j = df_k;
        d_j = -df_k;
        stopCond = false;
        nIterCG = 0;
        while (~stopCond & nIterCG < maxIter/10)
          if (d_j)'*B_k*d_j < 0
            if nIterCG == 0; p_k = d_j;
            else p_k = z_j; end;
            stopCond = true;
            eps_k = 0;
          end
          norm_r_j = r_j'*r_j;
          a_j = norm_r_j/(d_j'*B_k*d_j);
          z_j = z_j + a_j*d_j;
          r_j = r_j + a_j*B_k*d_j;
          if sqrt(r_j'*r_j) < eps_k; stopCond = true; end;
          b_j = r_j'*r_j/norm_r_j;
          d_j = -r_j + b_j*d_j;
          p_k = z_j;
          nIterCG = nIterCG + 1;
        end
      case 'gauss' % Gauss-Newton algorithm
      %================================= YOUR CODE HERE =============================
      p_k = (F.J(x_k)'*F.J(x_k)) \ (-F.J(x_k)'*F.r(x_k));
      %==============================================================================
    end
    
    % Call line search given by handle ls for computing step length
    alpha_k = ls(x_k, p_k, alpha0);
    
    % Update x_k and f_k
    x_k_1 = x_k;
    x_k = x_k + alpha_k*p_k;

    % Compute relative error norm
    normError = norm(x_k - x_k_1)/norm(x_k_1); 
    
    % Store iteration info
    info.xs = [info.xs x_k];
    info.alphas = [info.alphas alpha_k];
    
end

% Assign output values 
xMin = x_k;
fMin = F.f(x_k); 

