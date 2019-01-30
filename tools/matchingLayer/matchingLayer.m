function [qx, qy, angle2] = matchingLayer(px, py, angle1, c1, c2, thickness)
    angle2 = asin(sin(angle1) / c1 * c2);
    qx = px + tan(angle2)*thickness;
    qy = py + thickness;
end