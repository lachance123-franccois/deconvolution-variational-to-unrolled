function s = snr_db(xref, xest)
s = 10 * log10(sum(xref.^2, 1) ./ sum((xref - xest).^2, 1));
end
