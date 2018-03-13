function phi_matrix = getPhi(x,t)
phi_matrix = zeros(size(t,2),1);
for i=1:size(t,2)
    phi_matrix(i) = (x(1) + x(2)*t(i)^2)*exp(-x(3)*t(i));
end
end