function ops = build_ops(h, N)
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
