function x = gen_signal(type, N, opts)
if nargin < 3, opts = struct(); end
switch lower(type)
    case 'piecewise'
        nj = randi([opt_get(opts, 'min_jumps', 3), opt_get(opts, 'max_jumps', 10)]);
        pos = sort(randperm(N - 1, nj));
        levels = 2 * rand(nj + 1, 1) - 1;
        edges = [0, pos, N];
        x = zeros(N, 1);
        for k = 1:nj + 1
            x(edges(k) + 1:edges(k + 1)) = levels(k);
        end
    case 'piecewise_smooth'
        x = gen_signal('piecewise', N, opts);
        t = (0:N - 1)' / N;
        x = x + 0.3 * sin(2 * pi * (1 + 2 * rand) * t + 2 * pi * rand);
    case 'audio'
        S = load('handel.mat');
        s = S.y(:);
        i0 = randi(numel(s) - N);
        x = s(i0:i0 + N - 1);
        x = x / (max(abs(x)) + eps);
    otherwise
        error('gen_signal:type', 'Type de signal inconnu : %s', type);
end
end
