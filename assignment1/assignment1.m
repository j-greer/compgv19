%%
%% *ASSIGNMENT 1: EXERCISE 1*
%%
% Part A
close all; clear all

f = @(x,y) 2.*x + 4.*y + x.^2 - 2.*y.^2;
n = 100;
x = linspace(-1,1,n+1);
y = x;
[X,Y] = meshgrid(x,y);
alpha = 10;
figure
[C,h] = contourf(alpha*X, alpha*Y, f(alpha*X,alpha*Y),20);
colorbar();
%%
% Part B
%%
% $f = 2x + 4y + x^2 -2y^2$
%
% $f = (x+1)^2 - 2(y-1)^2 + 1$
%
% $S_c = {(x+1)^2 - 2(y-1)^2 + 1 = c}$ 
%
% $c = 1$ then point $S_1 = {-1,1}$
%
% $c \neq 1$ $S_c = \frac{(x+1)^2}{c-1} - \frac{2(y-1)^2}{c-1} = 1$
%%
% When the constant c is set to 1, we have a single solution at -1,1. Else
% we have a family of hyperbolas dependant on the value of c.
% 
%%
% Part C
%%
% $f(x,y) = 2x + 4y x^2 - 2y^2$
%
% $\frac{\partial f}{\partial x} = 2 + 2x$
%
% $\frac{\partial f}{\partial y} = 4 - 4y$
%
% $\frac{\partial f}{\partial x} = 0$
% 
% $x = -1$
% 
% $\frac{\partial f}{\partial y} = 0$
% 
% $y = 1$
%
% $\frac{\partial^2 f}{\partial x^2} = 2$
%
% $\frac{\partial^2 f}{\partial y^2} = -4$
%
% $\frac{\partial^2 f}{\partial x \partial y} = 0$
%
% $\frac{\partial^2 f}{\partial y \partial x} = 0$

%%
% This point (-1,1) is a stationary point but neither a maxima or minima.
% This point is an saddle point has described by the hessian
% 
%%
%% *ASSIGNMENT 1: EXERCISE 2*
%%
% Part D
%
% $A = B^TB = x^TB^TBx = (Bx)^T(Bx) \geq 0$
%%
% Therefore since this equality holds for all B, A is positive
% semi-definite
%
% s.t. $Ax = \lamda x$ non-negative eigenvalues
% $x^TAx = \lamda x^Tx$
% $\lamda = \frac{x^TAx}{x^Tx}$
% $A = B^TB$
% $\lamda = \frac{x^TB^TBx}{x^Tx}$
% $\lamda = \frac{\norm{Bx}^2}{\norm{x}^2} \geq 0$
% All eigenvalues are non-negative therefore A is postive definite

%%
% Part E
%
% $f(y+\alpha (x-y)) - \alpha f(x) - (1-\alpha)f(y) \leq 0$
%
% If this holds true, then the f(x) is convex.
%
% $(y^T + (\alpha x^T) - (\alpha y)^T)A(y + \alpha x - \alpha y) - \alpha x^TAx - y^TAy + \alpha y^TAy \leq 0$
%
% $\alpha(\alpha-1)(x-y)^TQ(x-y)$
%
% Since alpha is between 0 and 1 and Q is positive semidefinite, the entire term is leq to 0
