function x = tikhonov_deconv(y, ops, lambda, reg)
%TIKHONOV_DECONV  x = argmin ||y - Hx||^2 + lambda ||Rx||^2
%   Solution explicite x = (H'H + lambda R'R)^{-1} H'y, calculée en
%   O(N log N) car H et R sont circulantes.
%   reg : 'identity' | 'gradient' (défaut) | 'laplacian'
if nargin < 4, reg = 'gradient'; end
R2 = reg_spectrum(ops, reg);
x = real(ifft(conj(ops.Hf) .* fft(y) ./ (abs(ops.Hf).^2 + lambda * R2)));
end
