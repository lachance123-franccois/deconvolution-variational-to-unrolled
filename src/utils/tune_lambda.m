function [best, scores] = tune_lambda(solver, Y, X, grid)
%TUNE_LAMBDA Choisit l'hyperparamètre qui maximise le SNR moyen sur un
%   jeu de validation.
%   solver : @(Y, p) -> Xhat (N x B), p étant l'hyperparamètre testé
%   grid   : valeurs candidates
scores = zeros(numel(grid), 1);
for i = 1:numel(grid)
    Xh = solver(Y, grid(i));
    scores(i) = mean(snr_db(X, Xh));
end
[~, k] = max(scores);
best = grid(k);
end
