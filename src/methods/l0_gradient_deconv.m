function x = l0_gradient_deconv(y, ops, lambda, opts)
%L0_GRADIENT_DECONV Déconvolution avec pénalité L0 sur le gradient.
%   min_x 1/2 ||Hx - y||^2 + lambda ||Dx||_0
%   Problème non convexe, résolu par splitting semi-quadratique
%   (Xu, Zheng, Jia, 2013) avec beta croissant :
%     z <- Dx .* (|Dx|^2 >= lambda/beta)                (seuillage dur)
%     x <- (H'H + beta D'D)^{-1} (H'y + beta D'z)       (exact, par FFT)
%   Contrairement à la TV, la pénalité L0 favorise un vrai saut plutôt
%   qu'une rampe : elle est donc adaptée à l'estimation du noyau en
%   déconvolution aveugle.
%   opts : beta_max (1e5), kappa (2, facteur d'augmentation de beta)
if nargin < 4, opts = struct(); end
beta_max = opt_get(opts, 'beta_max', 1e5);
kappa    = opt_get(opts, 'kappa', 2);
Hf = ops.Hf; Df = ops.Df;
HtY = conj(Hf) .* fft(y);
beta = 2 * lambda;
x = real(ifft(HtY ./ (abs(Hf).^2 + beta * abs(Df).^2)));
while beta < beta_max
    Dx = ops.D(x);
    z = Dx .* (Dx.^2 >= lambda / beta);
    x = real(ifft((HtY + beta * conj(Df) .* fft(z)) ./ (abs(Hf).^2 + beta * abs(Df).^2)));
    beta = beta * kappa;
end
end
