function S = shift_matrices(N)
offs = [-1 0 1];
I = eye(N);
S = cell(1, 3);
for j = 1:3
    S{j} = circshift(I, -offs(j), 1);
end
end
