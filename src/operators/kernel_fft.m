function Hf = kernel_fft(h, N)
L = numel(h);
if L > N, error('kernel_fft:size', 'Noyau plus long que le signal.'); end
hp = zeros(N, 1);
hp(1:L) = h(:);
hp = circshift(hp, -(L - 1) / 2);
Hf = fft(hp);
end
