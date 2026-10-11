function x = naive_inverse(y, ops)
%NAIVE_INVERSE Inversion directe dans le domaine de Fourier : X = Y ./ H.
%   Illustre le caractère mal posé : le bruit est amplifié d'un facteur
%   1/|H(f)| aux fréquences où le noyau atténue fortement.
Hf = ops.Hf;
small = abs(Hf) < 1e-14;
Hf(small) = 1e-14;          % évite la division par zéro exacte
x = real(ifft(fft(y) ./ Hf));
end
