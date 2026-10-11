function [y, sigma, Hx] = degrade(x, ops, bsnr_db)
    % Cette fonction Applique le modele direct y = Hx + n, avec un bruit blanc gaussien.
    %   Le niveau de bruit est fixé par le BSNR (Blurred SNR) en dB :
    %       sigma^2 = var(Hx) / 10^(BSNR/10), calcule colonne par colonne.
    %   x peut etre une matrice N x B (une colonne par signal).
    Hx = ops.H(x);
    p = mean((Hx - mean(Hx, 1)).^2, 1);
    sigma = sqrt(p / 10^(bsnr_db / 10));
    y = Hx + sigma .* randn(size(Hx));
end
