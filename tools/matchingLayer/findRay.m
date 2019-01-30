function [endX, endY, r, time] = findRay(d, c, c_t, p_x, p_y)
    f = @(a) (computeRayDiff(a, d, c, c_t, p_x, p_y));
    %alpha = fminsearch(f, 0);
    opts = optimoptions('fmincon', 'Display', 'off');
    alpha = fmincon(f, 0, [], [], [], [], 0, 89/180*pi, [], opts);
    [endX, endY, r, time] = computeRay(alpha, d, c, c_t, p_y);
end