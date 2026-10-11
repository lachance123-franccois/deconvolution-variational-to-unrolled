function x = tv_denoise(v, lambda, nit)
%TV_DENOISE Débruitage TV 1D = opérateur proximal de lambda ||D.||_1.
%   x = argmin 1/2 ||x - v||^2 + lambda ||Dx||_1
%   Gradient projeté sur le problème dual (Chambolle, 2004) :
%   min_{|q| <= lambda} 1/2 ||v - D'q||^2,  puis x = v - D'q.
%   Opère sur les colonnes de v. nit : nombre d'itérations (100).
if nargin < 3, nit = 100; end
D  = @(u) circshift(u, -1, 1) - u;     % (Du)_i  = u_{i+1} - u_i
Dt = @(q) circshift(q, 1, 1) - q;      % (D'q)_i = q_{i-1} - q_i
q = zeros(size(v));
step = 0.25;                           % 1 / ||D||^2
for k = 1:nit
    q = q + step * D(v - Dt(q));
    q = min(max(q, -lambda), lambda);
end
x = v - Dt(q);
end
