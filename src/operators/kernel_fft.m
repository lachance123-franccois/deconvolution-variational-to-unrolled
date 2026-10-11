function Hf = kernel_fft(h, N)
%Fonction de transfert (TFD sur N points) d'un noyau centré.
%   Le coefficient central de h est placé à l'indice 1 (décalage nul), ce
%   qui correspond à une convolution circulaire sans retard :
%       (h * x)_i = sum_t h_t x_{i-t},   t = -(L-1)/2 ... (L-1)/2
L = numel(h);
if L > N, error('kernel_fft:size', 'Noyau plus long que le signal.'); end
hp = zeros(N, 1);
hp(1:L) = h(:);
hp = circshift(hp, -(L - 1) / 2);
Hf = fft(hp);
end
