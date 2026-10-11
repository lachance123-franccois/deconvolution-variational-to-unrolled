function [x, info] = admm_tv(y, ops, lambda, opts)
%ADMM_TV Déconvolution TV par ADMM.
%   min_x 1/2 ||Hx - y||^2 + lambda ||Dx||_1
%   Variable auxiliaire z = Dx, variable duale (mise à l'échelle) u :
%     x <- (H'H + rho D'D)^{-1} (H'y + rho D'(z - u))   [exact, par FFT]
%     z <- soft(Dx + u, lambda/rho)
%     u <- u + Dx - z
%   opts : rho (0.1), maxit (300), tol (1e-6), x0 (H'y), xtrue (suivi du SNR)
if nargin < 4, opts = struct(); end
rho   = opt_get(opts, 'rho', 0.1);
maxit = opt_get(opts, 'maxit', 300);
tol   = opt_get(opts, 'tol', 1e-6);
xtrue = opt_get(opts, 'xtrue', []);
x     = opt_get(opts, 'x0', ops.Ht(y));

Hf = ops.Hf; Df = ops.Df;
HtY = conj(Hf) .* fft(y);
den = abs(Hf).^2 + rho * abs(Df).^2;
z = ops.D(x);
u = zeros(size(z));
n = sqrt(numel(x));

info.obj = zeros(maxit, 1);
info.snr = nan(maxit, 1);
for k = 1:maxit
    x  = real(ifft((HtY + rho * conj(Df) .* fft(z - u)) ./ den));
    Dx = ops.D(x);
    zold = z;
    z  = soft_threshold(Dx + u, lambda / rho);
    u  = u + Dx - z;

    info.obj(k) = tv_objective(x, y, ops, lambda);
    if ~isempty(xtrue), info.snr(k) = mean(snr_db(xtrue, x)); end

    r_primal = norm(Dx(:) - z(:));
    r_dual   = rho * norm(reshape(ops.Dt(z - zold), [], 1));
    if r_primal < tol * n && r_dual < tol * n, break; end
end
info.iter = k;
info.obj = info.obj(1:k);
info.snr = info.snr(1:k);
end
