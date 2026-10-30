function [best, scores] = tune_lambda(solver, Y, X, grid)
scores = zeros(numel(grid), 1);
for i = 1:numel(grid)
    Xh = solver(Y, grid(i));
    scores(i) = mean(snr_db(X, Xh));
end
[~, k] = max(scores);
best = grid(k);
end
