function [x, h, info] = blind_deconv_am(y, L, opts)
%BLIND_DECONV_AM Déconvolution aveugle par minimisation alternée.
%   Phase 1 - estimation du noyau, en alternant :
%     - étape x : a priori parcimonieux sur le gradient (L0 par défaut,
%                 ou TV) avec le noyau courant
%     - étape h : min_h 1/2||x * h - y||^2 + mu/2 ||D h||^2
%                 s.c. h >= 0, sum(h) = 1  (gradient projeté)
%     avec une continuation sur lambda (forte -> faible).
%   Phase 2 - déconvolution non aveugle finale (ADMM-TV) avec le noyau estimé.
%
%   Pourquoi L0 plutôt que TV pour la phase 1 ? En 1D, une rampe et un
%   saut de même amplitude ont la même variation totale : la TV ne pousse
%   pas x vers des fronts raides, et le noyau converge vers un Dirac
%   (solution triviale x = y). La pénalité L0 compte les sauts et favorise
%   donc les fronts raides, ce qui force h à expliquer le flou.
%
%   opts : prior ('l0' | 'tv'), lambda (1e-3 pour L0, 0.02 pour TV),
%          lambda0 (10*lambda), outer (30), mu (0.1),
%          lambda_final (lambda TV de la phase 2, 0.02),
%          h0 (gaussienne sigma = 1), xtrue, htrue (suivi des erreurs)
if nargin < 3, opts = struct(); end
prior = lower(opt_get(opts, 'prior', 'l0'));
if strcmp(prior, 'l0'), lam_def = 1e-3; else, lam_def = 0.02; end
lam   = opt_get(opts, 'lambda', lam_def);
lam0  = opt_get(opts, 'lambda0', 10 * lam);
nout  = opt_get(opts, 'outer', 30);
mu    = opt_get(opts, 'mu', 0.1);
lam_f = opt_get(opts, 'lambda_final', 0.02);
h     = opt_get(opts, 'h0', make_kernel('gaussian', 1, L));
xtrue = opt_get(opts, 'xtrue', []);
htrue = opt_get(opts, 'htrue', []);
N = numel(y);
h = h(:);

lams = logspace(log10(lam0), log10(lam), nout);
x = y;
info.snr = nan(nout, 1);
info.kerr = nan(nout, 1);
info.h0 = h;
for t = 1:nout
    ops = build_ops(h, N);
    switch prior
        case 'l0'
            x = l0_gradient_deconv(y, ops, lams(t));
        case 'tv'
            x = admm_tv(y, ops, lams(t), struct('maxit', 100, 'x0', x));
        otherwise
            error('blind_deconv_am:prior', 'A priori inconnu : %s', prior);
    end
    h = update_kernel(y, x, h, mu, 50);
    if ~isempty(xtrue), info.snr(t) = snr_db(xtrue, x); end
    if ~isempty(htrue), info.kerr(t) = norm(h - htrue(:)) / norm(htrue); end
end
x = admm_tv(y, build_ops(h, N), lam_f, struct('maxit', 300));
info.lambdas = lams;
end
