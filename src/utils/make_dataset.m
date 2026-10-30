function [X, Y, sigma] = make_dataset(M, N, ops, bsnr_db, types, opts)
    if nargin < 6, opts = struct(); end
    if ~iscell(types), types = {types}; end
    X = zeros(N, M);
    for m = 1:M
        X(:, m) = gen_signal(types{randi(numel(types))}, N, opts);
    end
    [Y, sigma] = degrade(X, ops, bsnr_db);
end
