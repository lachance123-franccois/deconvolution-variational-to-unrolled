function T = exp02_linear_methods(cfg)
%Cette experience implemente les methodes de Wiener et Tikhonov avec un choix automatique de lambda.
%   Compare Tikhonov (identité / gradient) avec lambda choisi par :
%   oracle (meilleur SNR), GCV, courbe en L, principe de Morozov ;
%   et le filtre de Wiener oracle (spectre du vrai signal).
%   Partie 1 : un signal (figures). Partie 2 : moyenne sur 50 signaux.
if nargin < 1, cfg = default_config(); end
fprintf('\n=== Expérience 2 : méthodes linéaires ===\n');
rng(cfg.seed);
N = cfg.N;
ops = build_ops(make_kernel(cfg.kernel_type, cfg.kernel_param, cfg.kernel_len), N);
x = gen_signal(cfg.signal_type, N);
[y, sigma] = degrade(x, ops, cfg.bsnr);
lambdas = logspace(-6, 2, 200);
regs = {'identity', 'gradient'};
crit = {'gcv', 'lcurve', 'discrepancy'};
crit_names = {'GCV', 'Courbe en L', 'Morozov'};

% Partie 1 : un signal
rows = cell(0, 4);
curves = zeros(numel(lambdas), 2);
infos = struct();
for r = 1:2
    curves(:, r) = tik_snr_curve(x, y, ops, lambdas, regs{r});
    [s_or, k_or] = max(curves(:, r));
    rows(end + 1, :) = {['Tikhonov-' regs{r}], 'oracle', lambdas(k_or), s_or}; %#ok<AGROW>
    for c = 1:numel(crit)
        [lam, info] = select_lambda(y, ops, crit{c}, ...
            struct('reg', regs{r}, 'lambdas', lambdas, 'sigma', sigma));
        infos.(regs{r}).(crit{c}) = info;
        rows(end + 1, :) = {['Tikhonov-' regs{r}], crit_names{c}, lam, ...
            snr_db(x, tikhonov_deconv(y, ops, lam, regs{r}))}; %#ok<AGROW>
    end
end
nsr = sigma^2 ./ (abs(fft(x)).^2 / N);
xw = wiener_deconv(y, ops, nsr);
rows(end + 1, :) = {'Wiener', 'oracle (spectre vrai)', NaN, snr_db(x, xw)};
T = cell2table(rows, 'VariableNames', {'Methode', 'Choix', 'lambda', 'SNR_dB'});
disp(T);
writetable(T, fullfile(cfg.results_dir, 'exp02_un_signal.csv'));

% Figure : SNR en fonction de lambda
fig = figure('Name', 'Exp2 - SNR vs lambda', 'Position', [100 100 750 450]);
mk = {'o', 's', 'd'};
cols = lines(2);
for r = 1:2
    semilogx(lambdas, curves(:, r), 'Color', cols(r, :), 'LineWidth', 1.5); hold on;
end
for r = 1:2
    for c = 1:numel(crit)
        k = infos.(regs{r}).(crit{c}).index;
        semilogx(lambdas(k), curves(k, r), mk{c}, 'Color', cols(r, :), ...
            'MarkerFaceColor', cols(r, :), 'MarkerSize', 8, 'HandleVisibility', 'off');
    end
end
grid on; xlabel('\lambda'); ylabel('SNR (dB)');
legend('R = I', 'R = D (gradient)', 'Location', 'south');
title('Tikhonov : rond = GCV, carré = courbe en L, losange = Morozov');
save_figure(fig, cfg, 'exp02_snr_vs_lambda');

% Figure : courbe en L et GCV
ig = infos.gradient;
fig = figure('Name', 'Exp2 - Courbe en L et GCV', 'Position', [100 100 1000 400]);
subplot(1, 2, 1);
loglog(ig.lcurve.residual, ig.lcurve.seminorm, 'LineWidth', 1.5); hold on;
for c = 1:numel(crit)
    k = ig.(crit{c}).index;
    loglog(ig.lcurve.residual(k), ig.lcurve.seminorm(k), mk{c}, 'MarkerSize', 9, 'LineWidth', 1.5);
end
grid on; xlabel('||y - Hx_\lambda||^2'); ylabel('||Dx_\lambda||^2');
legend([{'Courbe en L'}, crit_names], 'Location', 'northeast'); title('Courbe en L');
subplot(1, 2, 2);
loglog(lambdas, ig.gcv.curve, 'LineWidth', 1.5); hold on;
loglog(lambdas(ig.gcv.index), ig.gcv.curve(ig.gcv.index), 'o', 'MarkerSize', 9, 'LineWidth', 1.5);
grid on; xlabel('\lambda'); ylabel('GCV(\lambda)'); title('Validation croisée généralisée');
save_figure(fig, cfg, 'exp02_courbe_L_gcv');

% Figure : reconstructions
lam_id = infos.identity.gcv.lambdas(infos.identity.gcv.index);
lam_gr = infos.gradient.gcv.lambdas(infos.gradient.gcv.index);
fig = figure('Name', 'Exp2 - Reconstructions', 'Position', [100 100 900 500]);
plot(x, 'k', 'LineWidth', 2); hold on;
plot(xw, 'LineWidth', 1.2);
plot(tikhonov_deconv(y, ops, lam_id, 'identity'), 'LineWidth', 1.2);
plot(tikhonov_deconv(y, ops, lam_gr, 'gradient'), 'LineWidth', 1.2);
grid on; xlabel('n');
legend('x (vrai)', 'Wiener oracle', 'Tikhonov R = I (GCV)', 'Tikhonov R = D (GCV)');
title('Méthodes linéaires : les discontinuités sont lissées');
save_figure(fig, cfg, 'exp02_reconstructions');

% Partie 2 : statistiques sur 50 signaux
nmc = 50;
R = zeros(nmc, 5);
for m = 1:nmc
    xm = gen_signal(cfg.signal_type, N);
    [ym, sm] = degrade(xm, ops, cfg.bsnr);
    R(m, 1) = max(tik_snr_curve(xm, ym, ops, lambdas, 'gradient'));
    for c = 1:numel(crit)
        l = select_lambda(ym, ops, crit{c}, struct('reg', 'gradient', 'lambdas', lambdas, 'sigma', sm, 'warn', false));
        R(m, 1 + c) = snr_db(xm, tikhonov_deconv(ym, ops, l, 'gradient'));
    end
    R(m, 5) = snr_db(xm, wiener_deconv(ym, ops, sm^2 ./ (abs(fft(xm)).^2 / N)));
end
names = {'Tikhonov-D oracle'; 'Tikhonov-D GCV'; 'Tikhonov-D courbe en L'; ...
         'Tikhonov-D Morozov'; 'Wiener oracle'};
T2 = table(names, mean(R)', std(R)', 'VariableNames', {'Methode', 'SNR_moyen_dB', 'Ecart_type'});
fprintf('Moyenne sur %d signaux :\n', nmc);
disp(T2);
writetable(T2, fullfile(cfg.results_dir, 'exp02_statistiques.csv'));

% Comparaison appariée : perte par rapport à l'oracle
loss = R(:, 1) - R(:, 2:4);
crit_lbl = {'GCV'; 'Courbe en L'; 'Morozov'};
T3 = table(crit_lbl, mean(loss)', std(loss)', median(loss)', ...
    'VariableNames', {'Critere', 'Perte_moy_dB', 'Ecart_type', 'Perte_mediane_dB'});
fprintf('Perte de SNR par rapport à l''oracle :\n');
disp(T3);
writetable(T3, fullfile(cfg.results_dir, 'exp02_perte_vs_oracle.csv'));

% Comparaison appariée : critères deux à deux
pairs = [1 2; 1 3; 2 3];
np = size(pairs, 1);
lbl = cell(np, 1); dm = zeros(np, 1); ci = zeros(np, 2); win = zeros(np, 1);
for i = 1:np
    a = pairs(i, 1); b = pairs(i, 2);
    d = R(:, 1 + a) - R(:, 1 + b);
    lbl{i} = sprintf('%s - %s', crit_lbl{a}, crit_lbl{b});
    dm(i) = mean(d);
    half = 1.96 * std(d) / sqrt(nmc);
    ci(i, :) = [dm(i) - half, dm(i) + half];
    win(i) = mean(d > 0);
end
T4 = table(lbl, dm, ci(:, 1), ci(:, 2), win, 'VariableNames', ...
    {'Comparaison', 'Diff_moy_dB', 'IC95_bas', 'IC95_haut', 'Part_victoires_1er'});
fprintf('Comparaison appariée des critères :\n');
disp(T4);
writetable(T4, fullfile(cfg.results_dir, 'exp02_comparaison_appariee.csv'));

% Figure : perte par rapport à l'oracle
fig = figure('Name', 'Exp2 - Perte vs oracle', 'Position', [100 100 600 400]);
for j = 1:3
    xj = j + 0.15 * (rand(nmc, 1) - 0.5);
    plot(xj, loss(:, j), '.', 'MarkerSize', 12); hold on;
    plot(j + [-0.25 0.25], mean(loss(:, j)) * [1 1], 'k-', 'LineWidth', 2);
end
set(gca, 'XTick', 1:3, 'XTickLabel', crit_lbl); xlim([0.5 3.5]);
grid on; ylabel('SNR oracle - SNR critère (dB)');
title(sprintf('Coût du choix automatique de lambda (%d signaux)', nmc));
save_figure(fig, cfg, 'exp02_perte_vs_oracle');
end

function s = tik_snr_curve(x, y, ops, lambdas, reg)
s = zeros(numel(lambdas), 1);
for i = 1:numel(lambdas)
    s(i) = snr_db(x, tikhonov_deconv(y, ops, lambdas(i), reg));
end
end