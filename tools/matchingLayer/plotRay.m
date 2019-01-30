function endX = plotRay(alpha, d, c, c_t, p_y)
    [endX, endY, r] = computeRay(alpha, d, c, c_t, p_y);
    plot([0; r(:,1); endX], [0; r(:,2); endY]);
end