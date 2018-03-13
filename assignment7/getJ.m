function J_matrix = getJ(x,t)
J_matrix = zeros(size(t,2),3);
for i=1:size(t,2)
    J_matrix(i,:) = [exp(-x(3)*t(i))
                     t(i)*exp(-x(3)*t(i))
                     -x(1)*t(i)*exp(-x(3)*t(i)) - x(2)*t(i)^2*exp(-x(3)*t(i))];
end
end