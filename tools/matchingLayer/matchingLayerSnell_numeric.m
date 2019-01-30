clc;
clear;
close all;

p_x = 25;
p_y = 80;

n = 2;
c = [3720; 2660; 1540]' * 1000;
d = [8.9e-05; 7.6e-05] * 1000;
% n = 4;
% d = ones(n,1)*0.5;
% c = linspace(4000, 1540, n + 1) * 1000;
c_t = c(end);
c = c(1:(end-1));

% compute some ray end positions
angles = linspace(0, 85 / 180 * pi, 1000);
endXs = zeros(1, length(angles));
for i = 1:length(angles)
    angle = angles(i);
    [endX, endY, ~] = computeRay(angle, d, c, c_t, p_y);
    endXs(i) = endX;
end

% compute delay difference w.r.t. no matching layers
[xs, ys] = meshgrid(0:p_x, sum(d):p_y);
ToF = zeros(size(xs));
ToF_uncorr = ToF;
for indY = 1:size(ys, 1)
    for indX = 1:size(xs, 2)
        x = xs(indY, indX);
        y = ys(indY, indX);
        
        [~, ~, ~, time] = findRay(d, c, c_t, x, y);
        ToF(indY, indX) = time;
        ToF_uncorr(indY, indX) = norm([x, y]) / c_t;
    end
end
F = 0.75;
freq = 7000000;
ApertureMask = xs <= (ys / (2*F));
ToF_error = (ToF_uncorr - ToF).* ApertureMask;
ToF_errorPeriods = ToF_error * freq;
ToF_errorRelative = (ToF_errorPeriods - ToF_errorPeriods(:,1)).* ApertureMask;

maxX = 0;
figure(1);
hold on;

% plot some rays
for angle = linspace(0, 80 / 180 * pi, 10)
    endX = plotRay(angle, d, c, c_t, p_y);
    maxX = max(maxX, endX);
end

layerY = 0;
for i = 1:n
    layerY = layerY + d(i);
    plot([-1, maxX + 1], [layerY, layerY], 'k-');
end

hold off;
xlabel('x');
ylabel('y');
axis ij

figure(2)
plot(angles/pi*180, endXs)
xlabel('alpha');
ylabel('x at target depth');

figure(3)
plot(angles/pi*180, abs(endXs - p_x))
xlabel('alpha');
ylabel('dist to p_x');

figure(4);
subplot(1,4,1);
imagesc(ToF_uncorr);
title('ToF uncorr');
subplot(1,4,2);
imagesc(ToF);
title('ToF true');
subplot(1,4,3);
imagesc(ToF_error);
title('ToF error');
subplot(1,4,4);
imagesc(ToF_errorRelative, [0, 1]);
title("ToF error (in periods) with F = " + F);
