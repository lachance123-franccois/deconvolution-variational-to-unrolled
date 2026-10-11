function x = haar_idwt(c, J)
%HAAR_IDWT Inverse (= adjointe) de haar_dwt.
N = size(c, 1);
x = c;
n = N / 2^J;
for j = 1:J
    s = x(1:n, :);
    d = x(n+1:2*n, :);
    a = zeros(2 * n, size(x, 2));
    a(1:2:2*n, :) = (s + d) / sqrt(2);
    a(2:2:2*n, :) = (s - d) / sqrt(2);
    x(1:2*n, :) = a;
    n = 2 * n;
end
end
