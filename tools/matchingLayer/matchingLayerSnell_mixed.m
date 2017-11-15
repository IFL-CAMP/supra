clc;
clear;
close all;

p_x = sym('p_x', 'positive');
p_y = sym('p_y');
assume(p_y >= 0)

n = 2;
c = [3720; 2660; 1540]' * 1000;
d = [8.9e-05; 7.6e-05] * 1000;
% n = 4;
% d = ones(n,1)*0.5;
% c = linspace(4000, 1540, n + 1) * 1000;
c_t = c(end);
c = c(1:(end-1));

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

sa = sym('sa');
assume(sa >= 0 & sa < 0.999);
[endX, endY, r, time] = computeRay(asin(sa), d, c, c_t, p_y);


endX
endXsimple = simplify(endX, 'IgnoreAnalyticConstraints',true)




time

timeSimple1 = simplify(time, 'Criterion','preferReal')


t = sym('t');
x = sym('x');
assume(t > 0 & x >= 0)

endXsimpleTaylor = taylor(endXsimple, sa, 'Order', 4)
saSol = simplify(solve(x == endXsimpleTaylor, sa, 'MaxDegree', 4))

[endXSol, endYSol, rSol, timeSol] = computeRay(asin(saSol), d, c, c_t, p_y);

% assume(p_y == 80)
% assume(x == 12)

% endXsimpleConstrained = simplify(endXsimple, 'IgnoreAnalyticConstraints',true, 'Criterion','preferReal')
% endXsimpleConstrainedTaylor = taylor(endXsimpleConstrained, sa)
% fplot([endXsimpleConstrained, endXsimpleConstrainedTaylor])

% goal = solve(12 == endXsimpleConstrainedTaylor, sa, 'IgnoreAnalyticConstraints',true)
% eval(goal)

% solve([t == timeSimple1, x == endXs% imple], x)

timeSol = simplify(timeSol, 'IgnoreAnalyticConstraints',true)
