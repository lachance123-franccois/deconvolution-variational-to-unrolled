function T = exp05_learned_methods(cfg)

if nargin < 1, cfg = default_config(); end
fprintf('\n=== Expérience 5 : réseau déroulé vs CNN boîte noire ===\n');
M = get_trained_models(cfg);
N = cfg.N;
ops = build_ops(make_kernel(cfg.kernel_type, cfg.kernel_param, cfg.kernel_len), N);
Hm = conv_matrix(ops.Hf);
S = shift_matrices(N);
rng(cfg.seed + 200);
[Xte, Yte] = make_dataset(cfg.n_test, N, ops, cfg.bsnr, cfg.train_types);
[Xau, Yau] = make_dataset(100, N, ops, cfg.bsnr, 'audio');

[names, fns, npar] = method_list(M, ops, Hm, S);
nm = numel(names);
m_in = zeros(nm, 1); s_in = zeros(nm, 1); m_out = zeros(nm, 1); tms = zeros(nm, 1);
Xh_all = cell(nm, 1);
for i = 1:nm
    tic; Xh = fns{i}(Yte); tms(i) = 1000 * toc / size(Yte, 2);
    v = snr_db(Xte, Xh);
    m_in(i) = mean(v); s_in(i) = std(v);
    m_out(i) = mean(snr_db(Xau, fns{i}(Yau)));
    Xh_all{i} = Xh;
end
T = table(names, m_in, s_in, m_out, tms, npar, 'VariableNames', ...
    {'Methode', 'SNR_test_dB', 'Ecart_type', 'SNR_audio_dB', 'Temps_ms_par_signal', 'Nb_parametres_appris'});
disp(T);
writetable(T, fullfile(cfg.results_dir, 'exp05_comparaison.csv'));

% Figures 

fig = figure('Name', 'Exp5 - Comparaison', 'Position', [100 100 800 420]);
bar(categorical(names, names), [m_in, m_out]); grid on;
ylabel('SNR moyen (dB)'); legend('Test (même distribution)', 'Audio (hors distribution)', 'Location', 'northwest');
title('Comparaison des méthodes');
save_figure(fig, cfg, 'exp05_barres');

k = 1;
fig = figure('Name', 'Exp5 - Exemple', 'Position', [100 100 950 500]);
plot(Xte(:, k), 'k', 'LineWidth', 2); hold on;
leg = {'x (vrai)'};
for i = 1:nm
    plot(Xh_all{i}(:, k), 'LineWidth', 1.1);
    leg{end + 1} = sprintf('%s (%.1f dB)', names{i}, snr_db(Xte(:, k), Xh_all{i}(:, k))); %#ok<AGROW>
end
legend(leg, 'Location', 'bestoutside'); grid on; xlabel('n');
title('Exemple de reconstruction (jeu de test)');
save_figure(fig, cfg, 'exp05_exemple');

if M.has_dl
    h = M.unrolled_hist;
    fig = figure('Name', 'Exp5 - Apprentissage du réseau déroulé', 'Position', [100 100 1000 380]);
    subplot(1, 2, 1);
    semilogy(h.train, 'Color', [0.6 0.6 0.6]); hold on;
    semilogy(movmean(h.train, 50), 'LineWidth', 1.5);
    grid on; xlabel('itération'); ylabel('perte (EQM)'); title('Perte d''apprentissage');
    subplot(1, 2, 2);
    plot(h.val, 'o-', 'LineWidth', 1.4); grid on;
    xlabel('époque'); ylabel('SNR validation (dB)'); title('Validation');
    save_figure(fig, cfg, 'exp05_apprentissage');

    P = M.unrolled;
    fig = figure('Name', 'Exp5 - Paramètres appris', 'Position', [100 100 1000 380]);
    subplot(1, 2, 1);
    semilogy(exp(P.logtau), 'o-', 'LineWidth', 1.3); hold on;
    semilogy(exp(P.logsig), 's-', 'LineWidth', 1.3);
    semilogy(exp(P.loglam), 'd-', 'LineWidth', 1.3);
    grid on; xlabel('couche k'); legend('\tau_k', '\sigma_k', '\lambda_k', 'Location', 'best');
    title('Pas et seuils appris par couche');
    subplot(1, 2, 2);
    plot(P.W', 'o-', 'LineWidth', 1.3); grid on;
    xlabel('couche k'); ylabel('coefficient');
    legend('w_{-1}', 'w_0', 'w_{+1}', 'Location', 'best');
    title('Filtres d''analyse appris (init. : [0 -1 1])');
    save_figure(fig, cfg, 'exp05_parametres_appris');
end
end

function [names, fns, npar] = method_list(M, ops, Hm, S)
names = {'Tikhonov-D'; 'ADMM-TV'};
fns = {@(Y) tikhonov_deconv(Y, ops, M.lam_tik, 'gradient'); ...
       @(Y) admm_tv(Y, ops, M.lam_tv, struct('maxit', 300))};
npar = [1; 1];
if M.has_dl
    den = M.denoiser;
    names = [names; {'PnP-ADMM (CNN)'; 'CP déroulé'; 'CNN boîte noire'}];
    fns = [fns; {@(Y) pnp_admm(Y, ops, @(v, ~) cnn_apply(den, v), struct('rho', M.rho_pnp, 'maxit', 50)); ...
                 @(Y) unrolled_cp_forward(M.unrolled, Y, Hm, S); ...
                 @(Y) cnn_apply(M.cnn, Y)}];
    npar = [npar; count_learnables(M.denoiser) + 1; ...
            sum(structfun(@numel, M.unrolled)); count_learnables(M.cnn)];
end
end


%EXP05_LEARNED_METHODS Methodes apprises vs methodes variationnelles.
%   Compare sur un jeu de test (même distribution que l'apprentissage)
%   et sur des extraits audio (hors distribution) :
%   Tikhonov, ADMM-TV, PnP-ADMM (CNN), Chambolle-Pock déroulé, CNN boîte noire.
%   Rapporte SNR, temps de calcul et nombre de paramètres appris.