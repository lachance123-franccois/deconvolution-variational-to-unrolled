function c = haar_dwt(x, J)
%HAAR_DWT Transformée en ondelettes de Haar orthonormée sur J niveaux.
%   Opère sur les colonnes de x (N divisible par 2^J).
%   Rangement : [approximation niveau J ; détails niveau J ; ... ; détails niveau 1]
N = size(x, 1);
if mod(N, 2^J) ~= 0
    error('haar_dwt:size', 'N doit être divisible par 2^J.');
end
c = x;
n = N;
for j = 1:J
    a = c(1:n, :);
    e = a(1:2:n, :);
    o = a(2:2:n, :);
    c(1:n/2, :)     = (e + o) / sqrt(2);
    c(n/2+1:n, :)   = (e - o) / sqrt(2);
    n = n / 2;
end
end
