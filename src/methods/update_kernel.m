function h = update_kernel(y, x, h, mu, nit)
%UPDATE_KERNEL Estimation du noyau à x fixé (déconvolution aveugle).
%   min_h 1/2 ||x * h - y||^2 + mu/2 ||D_L h||^2   s.c. h >= 0, sum(h) = 1
%   (support centré de longueur L = numel(h)), par gradient projeté.
L = numel(h);
N = numel(x);
t = -(L - 1) / 2:(L - 1) / 2;
Xc = zeros(N, L);
for j = 1:L
    Xc(:, j) = circshift(x, t(j));    % colonne j : x_{i - t_j}
end
Dl = diff(eye(L));
A = Xc' * Xc + mu * (Dl' * Dl);
b = Xc' * y;
Lip = norm(A);
h = h(:);
for k = 1:nit
    h = proj_simplex(h - (A * h - b) / Lip);
end
end
