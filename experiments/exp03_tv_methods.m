function T = exp03_tv_methods(cfg)
%EXP03_TV_METHODS Regularisation non lisse et algorithmes proximaux.
%   - TV par ADMM (rho = 0.01, 0.1, 1) et par Chambolle-Pock : convergence
%   - parcimonie en ondelettes de Haar par FISTA
%   - lambdas regles sur un jeu de validation (20 signaux)
%   - comparaison sur 50 signaux de test avec Tikhonov-gradient
if nargin < 1, cfg = default_config(); end
fprintf('\n=== Expérience 3 : TV et algorithmes proximaux ===\n');
rng(cfg.seed);
N = cfg.N;
ops = build_ops(make_kernel(cfg.kernel_type, cfg.kernel_param, cfg.kernel_len), N);

% Reglage des hyperparamètres sur validation 


[Xva, Yva] = make_dataset(20, N, ops, cfg.bsnr, cfg.signal_type);
grid_l = logspace(-3, -0.5, 20);
grid_t = logspace(-5, 1, 40);
[lam_tv, sc_tv]     = tune_lambda(@(Y, l) admm_tv(Y, ops, l, struct('maxit', 300)), Yva, Xva, grid_l);
[lam_haar, sc_haar] = tune_lambda(@(Y, l) fista_wavelet(Y, ops, l, struct('maxit', 300)), Yva, Xva, grid_l);
lam_tik             = tune_lambda(@(Y, l) tikhonov_deconv(Y, ops, l, 'gradient'), Yva, Xva, grid_t);
fprintf('lambda TV = %.3g, lambda Haar = %.3g, lambda Tikhonov = %.3g\n', lam_tv, lam_haar, lam_tik);

fig = figure('Name', 'Exp3 - Sensibilité à lambda', 'Position', [100 100 650 400]);
semilogx(grid_l, sc_tv, 'o-', grid_l, sc_haar, 's-', 'LineWidth', 1.4); grid on;
xlabel('\lambda'); ylabel('SNR moyen sur validation (dB)');
legend('TV (ADMM)', 'Haar (FISTA)', 'Location', 'south');
title('Sensibilité au paramètre de régularisation');
save_figure(fig, cfg, 'exp03_sensibilite_lambda');

% Convergence sur un signal 
x = gen_signal(cfg.signal_type, N);
y = degrade(x, ops, cfg.bsnr);
it = 500;
o = @(rho) struct('rho', rho, 'maxit', it, 'tol', 0, 'xtrue', x);
[~, iA01] = admm_tv(y, ops, lam_tv, o(0.01));
[xA, iA1] = admm_tv(y, ops, lam_tv, o(0.1));
[~, iA10] = admm_tv(y, ops, lam_tv, o(1));
[xC, iC]  = chambolle_pock_tv(y, ops, lam_tv, struct('maxit', it, 'tol', 0, 'xtrue', x));
[xH, iH]  = fista_wavelet(y, ops, lam_haar, struct('maxit', it, 'tol', 0, 'xtrue', x));
xref = chambolle_pock_tv(y, ops, lam_tv, struct('maxit', 20000, 'tol', 1e-12));
Fstar = min([iA01.obj; iA1.obj; iA10.obj; iC.obj; tv_objective(xref, y, ops, lam_tv)]);

fig = figure('Name', 'Exp3 - Convergence', 'Position', [100 100 1000 400]);
subplot(1, 2, 1);
semilogy(iA01.obj - Fstar + eps, 'LineWidth', 1.3); hold on;
semilogy(iA1.obj - Fstar + eps, 'LineWidth', 1.3);
semilogy(iA10.obj - Fstar + eps, 'LineWidth', 1.3);
semilogy(iC.obj - Fstar + eps, 'LineWidth', 1.3);
grid on; xlabel('itération'); ylabel('F(x_k) - F^*');
legend('ADMM \rho = 0.01', 'ADMM \rho = 0.1', 'ADMM \rho = 1', 'Chambolle-Pock');
title('Convergence vers le minimum du critère TV');
subplot(1, 2, 2);
plot(iA1.snr, 'LineWidth', 1.3); hold on;
plot(iC.snr, 'LineWidth', 1.3); plot(iH.snr, 'LineWidth', 1.3);
grid on; xlabel('itération'); ylabel('SNR (dB)');
legend('ADMM-TV (\rho = 0.1)', 'Chambolle-Pock TV', 'FISTA Haar', 'Location', 'southeast');
title('Qualité de reconstruction au fil des itérations');
save_figure(fig, cfg, 'exp03_convergence');

fig = figure('Name', 'Exp3 - Reconstructions', 'Position', [100 100 900 600]);
xT = tikhonov_deconv(y, ops, lam_tik, 'gradient');
subplot(2, 1, 1);
plot(x, 'k', 'LineWidth', 2); hold on; plot(y, 'Color', [0.7 0.7 0.7]);
plot(xT, 'LineWidth', 1.2);
legend('x', 'y', sprintf('Tikhonov-D (%.1f dB)', snr_db(x, xT))); grid on;
title('Méthode linéaire');
subplot(2, 1, 2);
plot(x, 'k', 'LineWidth', 2); hold on;
plot(xA, 'LineWidth', 1.2); plot(xH, 'LineWidth', 1.2);
legend('x', sprintf('ADMM-TV (%.1f dB)', snr_db(x, xA)), ...
    sprintf('FISTA Haar (%.1f dB)', snr_db(x, xH))); grid on;
title('Régularisations parcimonieuses : les fronts sont préservés'); xlabel('n');
save_figure(fig, cfg, 'exp03_reconstructions');
fprintf('Écart ADMM / Chambolle-Pock sur ce signal : %.2e (relatif)\n', norm(xA - xC) / norm(xA));

%  Statistiques sur 50 signaux de test 
[Xte, Yte] = make_dataset(50, N, ops, cfg.bsnr, cfg.signal_type);
names = {'Tikhonov-D'; 'ADMM-TV'; 'Chambolle-Pock TV'; 'FISTA Haar'};
solvers = {@(Y) tikhonov_deconv(Y, ops, lam_tik, 'gradient'), ...
           @(Y) admm_tv(Y, ops, lam_tv, struct('maxit', 300)), ...
           @(Y) chambolle_pock_tv(Y, ops, lam_tv, struct('maxit', 300)), ...
           @(Y) fista_wavelet(Y, ops, lam_haar, struct('maxit', 300))};
m = zeros(4, 1); s = zeros(4, 1); tms = zeros(4, 1);
for i = 1:4
    tic; Xh = solvers{i}(Yte); tms(i) = 1000 * toc / size(Yte, 2);
    v = snr_db(Xte, Xh);
    m(i) = mean(v); s(i) = std(v);
end
T = table(names, m, s, tms, 'VariableNames', {'Methode', 'SNR_moyen_dB', 'Ecart_type', 'Temps_ms_par_signal'});
disp(T);
writetable(T, fullfile(cfg.results_dir, 'exp03_statistiques.csv'));
end
