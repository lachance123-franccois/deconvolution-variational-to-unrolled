function X = unrolled_cp_forward(params, Y, Hm, S)
K = size(params.W, 2);
x = Hm' * Y;
xbar = x;
p = zeros(size(Y));
q = zeros(size(Y));
for k = 1:K
    tau = exp(params.logtau(k));
    sig = exp(params.logsig(k));
    lam = exp(params.loglam(k));
    w = params.W(:, k);

    Dxb = w(1) * (S{1} * xbar) + w(2) * (S{2} * xbar) + w(3) * (S{3} * xbar);
    p = (p + sig * (Hm * xbar - Y)) / (1 + sig);
    q = min(max(q + sig * Dxb, -lam), lam);
    Dtq = w(1) * (S{1}' * q) + w(2) * (S{2}' * q) + w(3) * (S{3}' * q);

    xnew = x - tau * (Hm' * p + Dtq);
    xbar = 2 * xnew - x;
    x = xnew;
end
X = x;
end
%Passe avant du Chambolle-Pock déroulé.
%   Problème sous-jacent :  min_x 1/2||Hx - y||^2 + lambda ||Dx||_1
%   écrit avec K = [H ; D] (aucune inversion de matrice) :
%     p    <- (p + sigma (H xbar - y)) / (1 + sigma)     prox de f1*
%     q    <- proj_[-lambda, lambda](q + sigma D xbar)   prox de f2*
%     x+   <- x - tau (H'p + D'q)
%     xbar <- 2 x+ - x
%   Chaque itération = une couche avec ses propres (tau, sigma, lambda, D).
%
%   params : structure (numérique ou dlarray), cf. unrolled_cp_init
%   Y      : observations N x B
%   Hm     : matrice de convolution N x N (conv_matrix)
%   S      : matrices de décalage (shift_matrices)
%   Compatible dlarray : seules des opérations différentiables sont utilisées.