% Numerical Optimisation: assignment 7
% Keshav Iyengar

clear variables;
close all;

phi = @(x,t) (x(1) + x(2)*t^2)*exp(-x(3)*t);
t = 0:4/200:4;
max_phi = 20.7146; %max value found
sigma = 0.05*max_phi;

measurements = zeros(1,size(t,2));
for i=1:size(t,2)
   x0 = [3 150 2];
   measurements(i) = phi(x0,t(i)) + mvnrnd(0,sigma);
end
