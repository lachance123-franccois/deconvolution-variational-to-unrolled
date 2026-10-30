function [params, hist] = train_unrolled_cp(Xtr, Ytr, Hm, Xval, Yval, opts)
K      = opt_get(opts, 'layers', 15);
epochs = opt_get(opts, 'epochs', 40);
bs     = opt_get(opts, 'batch', 32);
lr0    = opt_get(opts, 'lr', 1e-2);
lam0   = opt_get(opts, 'lam0', 0.03);

N = size(Xtr, 1);
M = size(Xtr, 2);
S = shift_matrices(N);
p = structfun(@dlarray, unrolled_cp_init(K, lam0), 'UniformOutput', false);
avgG = []; avgSq = [];
it = 0;
nb = floor(M / bs);
hist.train = zeros(epochs * nb, 1);
hist.val = zeros(epochs, 1);

for ep = 1:epochs
    lr = lr0 * 0.5^floor(3 * (ep - 1) / epochs);  
    idx = randperm(M);
    for b = 1:nb
        it = it + 1;
        sel = idx((b - 1) * bs + 1:b * bs);
        [loss, grad] = dlfeval(@unrolled_loss, p, Ytr(:, sel), Xtr(:, sel), Hm, S);
        [p, avgG, avgSq] = adamupdate(p, grad, avgG, avgSq, it, lr);
        hist.train(it) = double(extractdata(loss));
    end
    pv = structfun(@(v) double(extractdata(v)), p, 'UniformOutput', false);
    hist.val(ep) = mean(snr_db(Xval, unrolled_cp_forward(pv, Yval, Hm, S)));
end
params = structfun(@(v) double(extractdata(v)), p, 'UniformOutput', false);
end
