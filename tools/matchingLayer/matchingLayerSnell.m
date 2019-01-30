clc;
clear;
close all;

n = 4;

c_t = sym('c_t', 'real');
p_x = sym('px', 'positive');
p_y = sym('py');
d = sym('d', [n, 1], 'positive');
c = sym('c', [1, n], 'positive');

alpha = sym('alpha', [1, 1]);
assume(alpha >= 0 & alpha < 90);
alpha0 = alpha(1);

D = sum(d);
if n == 4
    assume(p_y > D & ...
        d > 0 & ...
        c(1) > c(2) & c(2) > c(3) & c(3) > c(4) & c(4) > c_t & c_t > 0);
end
if n == 2
    assume(p_y > D & ...
        d > 0 & ...
        c(1) > c(2) & c(2) > c_t & c_t > 0);
end

% r(1, :) = [tan(alpha)*d(1), d(1)];
% for i = 2:n
%     [qx, qy, angle2] = matchingLayer(r(i-1, 1), r(i-1, 2), alpha(i-1), c(i-1), c(i), d(i));
%     r(i, :) = [qx, qy];
%     alpha(i) = angle2;
% end
sa = sym('sa');
assume(sa >= 0 & sa < 1);
[endX, endY, r, time] = computeRay(asin(sa), d, c, c_t, p_y);

%solve(endX == p_x, alpha0)

endX
endXsimple = simplify(endX, 'IgnoreAnalyticConstraints',true)
endXgrad = diff(endXsimple, sa)

time

timeSimple1 = simplify(time)
timeSimple2 = simplify(time, 'Steps', 10, 'IgnoreAnalyticConstraints',true)
