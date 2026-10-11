function T = exp06_robustness(cfg)
%Robustesse et generalisation.
%   (a) niveau de bruit différent de celui de l'apprentissage
%   (b) erreur sur le noyau : vrai sigma_h = cfg.kernel_param, le modele
%       utilise sigma_m (les methodes à modele reçoivent le mauvais H)
%   (c) changement de noyau connu : les méthodes à modèle reçoivent le
%       bon H ; le CNN boîte noire, appris pour un seul noyau, ne peut pas
%       l'exploiter.
%   Tous les hyperparamètres restent ceux regles dans les conditions
%   nominales (pas de re-entraînement).

if nargin < 1, cfg = default_config(); end
fprintf('\n=== Expérience 6 : robustesse et généralisation ===\n');
M = get_trained_models(cfg);
N = cfg.N;
nt = 100;
S = shift_matrices(N);
mk = @(s) build_ops(make_kernel(cfg.kernel_type, s, cfg.kernel_len), N);
ops0 = mk(cfg.kernel_param);
names = {'Tikhonov-D', 'ADMM-TV', 'PnP-ADMM (CNN)', 'CP déroulé', 'CNN boîte noire'};
rng(cfg.seed + 400);

% (a) niveau de bruit 
bsnrs = [15 20 25 30 35 40];
Ra = zeros(numel(bsnrs), 5);
for i = 1:numel(bsnrs)
    [X, Y] = make_dataset(nt, N, ops0, bsnrs(i), cfg.train_types);
    Ra(i, :) = eval_methods(M, X, Y, ops0, S);
end

% (b) erreur sur le noyau
sig_m = cfg.kernel_param * [0.67 0.83 1 1.17 1.33];
[X, Y] = make_dataset(nt, N, ops0, cfg.bsnr, cfg.train_types);
Rb = zeros(numel(sig_m), 5);
for i = 1:numel(sig_m)
    Rb(i, :) = eval_methods(M, X, Y, mk(sig_m(i)), S);
end

% (c) changement de noyau (connu) 
sig_t = cfg.kernel_param * [0.5 0.67 1 1.33 1.67];
Rc = zeros(numel(sig_t), 5);
for i = 1:numel(sig_t)
    opsi = mk(sig_t(i));
    [X, Y] = make_dataset(nt, N, opsi, cfg.bsnr, cfg.train_types);
    Rc(i, :) = eval_methods(M, X, Y, opsi, S);
end

% Figures et tableaux 
fig = figure('Name', 'Exp6 - Robustesse', 'Position', [50 100 1400 400]);
subplot(1, 3, 1); plot_res(bsnrs, Ra, names);
xline(cfg.bsnr, ':k', 'HandleVisibility', 'off');
xlabel('BSNR de test (dB)'); title('(a) Niveau de bruit');
subplot(1, 3, 2); plot_res(sig_m, Rb, names);
xline(cfg.kernel_param, ':k', 'HandleVisibility', 'off');
xlabel('\sigma_h supposé par le modèle'); title(sprintf('(b) Erreur sur le noyau (vrai : %g)', cfg.kernel_param));
subplot(1, 3, 3); plot_res(sig_t, Rc, names);
xline(cfg.kernel_param, ':k', 'HandleVisibility', 'off');
xlabel('\sigma_h réel (connu des méthodes à modèle)'); title('(c) Nouveau noyau');
save_figure(fig, cfg, 'exp06_robustesse');

vn = matlab.lang.makeValidName(names);
T = struct();
T.bruit = array2table([bsnrs', Ra], 'VariableNames', [{'BSNR_dB'}, vn]);
T.erreur_noyau = array2table([sig_m', Rb], 'VariableNames', [{'sigma_modele'}, vn]);
T.nouveau_noyau = array2table([sig_t', Rc], 'VariableNames', [{'sigma_reel'}, vn]);
fn = fieldnames(T);
for i = 1:numel(fn)
    fprintf('--- %s ---\n', fn{i}); disp(T.(fn{i}));
    writetable(T.(fn{i}), fullfile(cfg.results_dir, ['exp06_' fn{i} '.csv']));
end
end

function r = eval_methods(M, X, Y, ops, S)
%EVAL_METHODS SNR moyen de chaque méthode ; ops = modèle utilisé (peut être faux).
r = nan(1, 5);
r(1) = mean(snr_db(X, tikhonov_deconv(Y, ops, M.lam_tik, 'gradient')));
r(2) = mean(snr_db(X, admm_tv(Y, ops, M.lam_tv, struct('maxit', 300))));
if M.has_dl
    den = M.denoiser;
    r(3) = mean(snr_db(X, pnp_admm(Y, ops, @(v, ~) cnn_apply(den, v), struct('rho', M.rho_pnp, 'maxit', 50))));
    r(4) = mean(snr_db(X, unrolled_cp_forward(M.unrolled, Y, conv_matrix(ops.Hf), S)));
    r(5) = mean(snr_db(X, cnn_apply(M.cnn, Y)));       % n'utilise pas ops
end
end

function plot_res(xv, R, names)
mk = {'o-', 's-', 'd-', '^-', 'v-'};
for j = 1:size(R, 2)
    if all(isnan(R(:, j))), continue; end
    plot(xv, R(:, j), mk{j}, 'LineWidth', 1.3, 'DisplayName', names{j}); hold on;
end
grid on; ylabel('SNR moyen (dB)'); legend('Location', 'best');
end
