function [X, Y, sigma] = make_dataset(M, N, ops, bsnr_db, types, opts)
    % Genere M paires (x, y) sous forme de matrices N x M.
    %   types : nom de type (cf. gen_signal) ou cellule de types ; dans ce
    %   dernier cas chaque signal tire un type au hasard.
    if nargin < 6, opts = struct(); end
    if ~iscell(types), types = {types}; end
    X = zeros(N, M);
    for m = 1:M
        X(:, m) = gen_signal(types{randi(numel(types))}, N, opts);
    end
    [Y, sigma] = degrade(X, ops, bsnr_db);
end
