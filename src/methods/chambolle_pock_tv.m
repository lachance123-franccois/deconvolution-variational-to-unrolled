function [x, info] = chambolle_pock_tv(y, ops, lambda, opts)
%CHAMBOLLE_POCK_TV Déconvolution TV par l'algorithme primal-dual
%   de Chambolle-Pock (2011).
%   min_x G(x) + F(Dx),  G(x) = 1/2||Hx - y||^2,  F(z) = lambda ||z||_1
%     p    <- proj_[-lambda, lambda](p + sigma D xbar)   (prox de F*)
%     x+   <- prox_{tau G}(x - tau D'p)                   (exact, par FFT)
%     xbar <- x+ + theta (x+ - x)
%   Convergence si tau * sigma * ||D||^2 < 1, avec ||D||^2 <= 4.
%   opts : tau (5), sigma (0.99/(4 tau)), theta (1), maxit, tol, x0, xtrue
if nargin < 4, opts = struct(); end
tau   = opt_get(opts, 'tau', 5);   % grand pas primal : le prox de G est exact
sigma = opt_get(opts, 'sigma', 0.99 / (4 * tau));
theta = opt_get(opts, 'theta', 1);
maxit = opt_get(opts, 'maxit', 300);
tol   = opt_get(opts, 'tol', 1e-6);
xtrue = opt_get(opts, 'xtrue', []);
x     = opt_get(opts, 'x0', ops.Ht(y));

Hf = ops.Hf;
tHtY = tau * conj(Hf) .* fft(y);
den = tau * abs(Hf).^2 + 1;
xbar = x;
p = zeros(size(x));

info.obj = zeros(maxit, 1);
info.snr = nan(maxit, 1);
for k = 1:maxit
    p = min(max(p + sigma * ops.D(xbar), -lambda), lambda);
    xold = x;
    v = x - tau * ops.Dt(p);
    x = real(ifft((tHtY + fft(v)) ./ den));
    xbar = x + theta * (x - xold);

    info.obj(k) = tv_objective(x, y, ops, lambda);
    if ~isempty(xtrue), info.snr(k) = mean(snr_db(xtrue, x)); end
    if norm(x(:) - xold(:)) < tol * max(norm(xold(:)), eps), break; end
end
info.iter = k;
info.obj = info.obj(1:k);
info.snr = info.snr(1:k);
end
