function M = conv_matrix(Tf)
N = numel(Tf);
M = real(ifft(Tf(:) .* fft(eye(N))));
end
