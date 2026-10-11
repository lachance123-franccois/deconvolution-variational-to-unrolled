function S = shift_matrices(N)
%SHIFT_MATRICES Matrices de décalage circulaire pour les décalages -1, 0, +1.
%   (S{j} x)_i = x_{i + s_j},  s = [-1 0 1].
%   Un filtre à 3 coefficients w s'écrit alors D = sum_j w(j) S{j} ;
%   w = [0 -1 1] redonne le gradient (Dx)_i = x_{i+1} - x_i.
offs = [-1 0 1];
I = eye(N);
S = cell(1, 3);
for j = 1:3
    S{j} = circshift(I, -offs(j), 1);
end
end
