function ops = build_ops(h, N)
%BUILD_OPS Opérateurs du modèle direct (conditions aux bords périodiques).
%   ops.H, ops.Ht : convolution par h et son adjoint
%   ops.D, ops.Dt : différences finies (Dx)_i = x_{i+1} - x_i et adjoint
%   ops.Hf, ops.Df: fonctions de transfert (les opérateurs sont circulants,
%                   donc diagonalisés par la TFD)
%   Toutes les fonctions acceptent une matrice N x B (une colonne par signal).
Hf = kernel_fft(h, N);
d = zeros(N, 1); d(1) = -1; d(N) = 1;
Df = fft(d);

ops.N  = N;
ops.h  = h(:);
ops.Hf = Hf;
ops.Df = Df;
ops.H  = @(x) real(ifft(Hf .* fft(x)));
ops.Ht = @(x) real(ifft(conj(Hf) .* fft(x)));
ops.D  = @(x) real(ifft(Df .* fft(x)));
ops.Dt = @(x) real(ifft(conj(Df) .* fft(x)));
end
