clc;
clear;
close all;

n = 4;

c_t = sym('c_t', 'real');
px = sym('px', 'positive');
py = sym('py');
r = sym('r', [1, 2], 'real');
d = sym('d', [n, 1], 'positive');
c = sym('c', [n, 1], 'positive');

alpha = sym('alpha', [1, 1]);
assume(alpha >= 0 & alpha < 90);
alpha0 = alpha(1);

D = sum(d);
assume(py > D);

r(1, :) = [tan(alpha)*d(1), d(1)];
for i = 2:n
    [qx, qy, angle2] = matchingLayer(r(i-1, 1), r(i-1, 2), alpha(i-1), c(i-1), c(i), d(i));
    r(i, :) = [qx, qy];
    alpha(i) = angle2;
end

[px_canididate, py_candidate, ~] = matchingLayer(r(n, 1), r(n, 2), alpha(n), c(n), c_t, py - r(n,2));

t=  symfun((1 / c(1) * sqrt(d(1)^2 + (0 - r(1))^2) + ...
    sum(1./c(2:n) .* sqrt(d(2:end).^2 + (r(1:(end-1)) - r(2:end)).^2)) + ...
    1 / c_t *sqrt((px - r(n))^2 + (py - D)^2)), r);

dt = gradient(t, r);

canditates = solve(dt == 0, r, 'ReturnConditions', true);