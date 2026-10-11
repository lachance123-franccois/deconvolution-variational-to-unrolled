function M = conv_matrix(Tf)
%Matrice circulante N x N associée à la fonction de transfert Tf.
%   Utile pour les méthodes apprises (dlarray) et pour l'analyse SVD.
N = numel(Tf);
M = real(ifft(Tf(:) .* fft(eye(N))));
end
