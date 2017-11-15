function [endX, endY, r, time] = computeRay(alpha, d, c, c_t, p_y)
    n = length(d);
    
    r(1, :) = [tan(alpha)*d(1), d(1)];
    for i = 2:n
        [qx, qy, angle2] = matchingLayer(r(i-1, 1), r(i-1, 2), alpha(i-1), c(i-1), c(i), d(i));
        r(i, :) = [qx, qy];
        alpha(i) = angle2;
    end

    [endX, endY, endAlpha] = matchingLayer(r(n, 1), r(n, 2), alpha(n), c(n), c_t, p_y - r(n,2));
    
    time = sum(permute(sqrt(sum(([r; endX, endY] - [0, 0; r]).^2, 2)), [2, 1]) ...
           ./ [c, c_t]);
end