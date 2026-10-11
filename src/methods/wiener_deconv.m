function x = wiener_deconv(y, ops, nsr)
%WIENER_DECONV Filtre de Wiener.
%   X(f) = conj(H(f)) / (|H(f)|^2 + S_n(f)/S_x(f)) * Y(f)
%   nsr : rapport S_n/S_x, scalaire ou vecteur N x 1.
%   Remarque : avec S_x(f) proportionnel à 1/|D(f)|^2 (spectre en 1/f^2,
%   typique des signaux constants par morceaux), on retrouve exactement
%   Tikhonov avec régularisation gradient.
G = conj(ops.Hf) ./ (abs(ops.Hf).^2 + nsr);
x = real(ifft(G .* fft(y)));
end
