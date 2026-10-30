function [y, sigma, Hx] = degrade(x, ops, bsnr_db)
    Hx = ops.H(x);
    p = mean((Hx - mean(Hx, 1)).^2, 1);
    sigma = sqrt(p / 10^(bsnr_db / 10));
    y = Hx + sigma .* randn(size(Hx));
end
