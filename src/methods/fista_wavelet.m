function [x, info] = fista_wavelet(y, ops, lambda, opts)
%FISTA_WAVELET Déconvolution avec a priori de parcimonie en ondelettes de Haar.
%   Formulation synthèse :  x = W' c,
%   min_c 1/2 ||H W' c - y||^2 + lambda ||c_details||_1
%   résolue par FISTA (Beck & Teboulle, 2009). Les coefficients
%   d'approximation ne sont pas pénalisés.
%   opts : levels (4), maxit (300), tol (1e-6), xtrue
if nargin < 4, opts = struct(); end
J     = opt_get(opts, 'levels', 4);
maxit = opt_get(opts, 'maxit', 300);
tol   = opt_get(opts, 'tol', 1e-6);
xtrue = opt_get(opts, 'xtrue', []);
N = ops.N;

Lip = max(abs(ops.Hf).^2);            % W orthonormée => ||H W'||^2 = max|H|^2
mask = true(N, 1);
mask(1:N / 2^J) = false;              % coefficients pénalisés

c = haar_dwt(ops.Ht(y), J);
v = c;
t = 1;
info.obj = zeros(maxit, 1);
info.snr = nan(maxit, 1);
for k = 1:maxit
    g = haar_dwt(ops.Ht(ops.H(haar_idwt(v, J)) - y), J);
    cnew = v - g / Lip;
    cnew(mask, :) = soft_threshold(cnew(mask, :), lambda / Lip);
    tnew = (1 + sqrt(1 + 4 * t^2)) / 2;
    v = cnew + ((t - 1) / tnew) * (cnew - c);
    dc = norm(cnew(:) - c(:)) / max(norm(c(:)), eps);
    c = cnew;
    t = tnew;

    x = haar_idwt(c, J);
    info.obj(k) = 0.5 * sum((ops.H(x) - y).^2, 'all') + lambda * sum(abs(c(mask, :)), 'all');
    if ~isempty(xtrue), info.snr(k) = mean(snr_db(xtrue, x)); end
    if dc < tol, break; end
end
info.iter = k;
info.obj = info.obj(1:k);
info.snr = info.snr(1:k);
end
