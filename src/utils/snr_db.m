function s = snr_db(xref, xest)
%SNR_DB Rapport signal sur bruit de reconstruction (dB), par colonne.
%   SNR = 10 log10( ||x||^2 / ||x - x_est||^2 )
s = 10 * log10(sum(xref.^2, 1) ./ sum((xref - xest).^2, 1));
end
