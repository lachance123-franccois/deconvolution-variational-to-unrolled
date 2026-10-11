function y = soft_threshold(v, t)
%SOFT_THRESHOLD Seuillage doux : opérateur proximal de t*||.||_1.
y = sign(v) .* max(abs(v) - t, 0);
end
