function err = computeRayDiff(alpha, d, c, c_t, p_x, p_y)
    endX = computeRay(alpha, d, c, c_t, p_y);
    err = abs(endX - p_x);
end