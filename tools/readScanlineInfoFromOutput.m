filename = 'C:\Users\goeblr\ownCloud\syncCode\scanlines_60_neg60.txt';
f = fopen(filename);

numscanlines = 256;
arrScale = 3;

nums = zeros(1,numscanlines);
firstX = zeros(1, numscanlines);
firstY = zeros(1, numscanlines);
lastX = zeros(1, numscanlines);
lastY = zeros(1,numscanlines);
positions = zeros(3, numscanlines);
directions = zeros(3, numscanlines);
for i = 1:numscanlines
%     scanline num 0
%     firstActiveElementIndex.x: 0
%     firstActiveElementIndex.y: 0
%     lastActiveElementIndex.x: 31
%     lastActiveElementIndex.y: 11
%     pos: 0, 0, -3
%     dir: -0.447214, 0.774597, -0.447214
    r = textscan(f,'%s %s %d',1,'Delimiter',' ');
    nums(i) = r{3};
    r = textscan(f,'%s %f',1,'Delimiter',' ');
    firstX(i) = r{2};
    r = textscan(f,'%s %f',1,'Delimiter',' ');
    firstY(i) = r{2};
    r = textscan(f,'%s %f',1,'Delimiter',' ');
    lastX(i) = r{2};
    r = textscan(f,'%s %f',1,'Delimiter',' ');
    lastY(i) = r{2};
    r = textscan(f,'%s %f, %f, %f',1,'Delimiter',' ');
    positions(:,i) = [r{2}; r{3};r{4}];
    r = textscan(f,'%s %f, %f, %f',1,'Delimiter',' ');
    directions(:,i) = [r{2}; r{3};r{4}];
end
scanlines = {nums, firstX, firstY, lastX, lastY, positions, directions};

fclose(f);

figure;
plot3(positions(1,:), positions(2,:), positions(3,:));
hold on;
quiver3(positions(1,:), positions(2,:), positions(3,:), arrScale*directions(1,:), arrScale*directions(2,:), arrScale*directions(3,:), 0);
hold off;
axis equal;