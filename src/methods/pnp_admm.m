function [x, info] = pnp_admm(y, ops, denoiser, opts)
%PNP_ADMM Plug-and-Play ADMM (Venkatakrishnan et al., 2013).
%   On part de l'ADMM pour  min 1/2||Hx - y||^2 + R(x)  avec z = x :
%     x <- (H'H + rho I)^{-1}(H'y + rho (z - u))   [exact, par FFT]
%     z <- prox_{R/rho}(x + u)    remplacé par un débruiteur quelconque
%     u <- u + x - z
%   denoiser : @(v, rho) -> z. Avec @(v,rho) tv_denoise(v, lambda/rho)
%              on retrouve exactement la TV ; avec un CNN, l'a priori est appris.
%   opts : rho (1), rho_growth (1 : rho constant), maxit (50), tol (1e-5), xtrue
if nargin < 4, opts = struct(); end
rho    = opt_get(opts, 'rho', 1);
growth = opt_get(opts, 'rho_growth', 1);
maxit  = opt_get(opts, 'maxit', 50);
tol    = opt_get(opts, 'tol', 1e-5);
xtrue  = opt_get(opts, 'xtrue', []);

Hf = ops.Hf;
HtY = conj(Hf) .* fft(y);
x = ops.Ht(y);
z = x;
u = zeros(size(x));
info.snr = nan(maxit, 1);
for k = 1:maxit
    x = real(ifft((HtY + rho * fft(z - u)) ./ (abs(Hf).^2 + rho)));
    zold = z;
    z = denoiser(x + u, rho);
    u = u + x - z;

    rho_new = rho * growth;
    u = u * rho / rho_new;            % variable duale mise à l'échelle
    rho = rho_new;

    if ~isempty(xtrue), info.snr(k) = mean(snr_db(xtrue, z)); end
    if norm(z(:) - zold(:)) < tol * max(norm(zold(:)), eps), break; end
end
x = z;
info.iter = k;
info.snr = info.snr(1:k);
end
