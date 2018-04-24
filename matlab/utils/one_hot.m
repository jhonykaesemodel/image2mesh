function y_one_hot = one_hot(y)

y_one_hot = zeros(size(y, 1), 30);
for i = 1:30 % TODO: 30 3D models in the graph
    rows = y == i;
    y_one_hot( rows, i ) = 1;
end
