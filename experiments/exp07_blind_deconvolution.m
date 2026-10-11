function T = exp07_blind_deconvolution(cfg)


if nargin < 1, cfg = default_config(); end
fprintf('\n=== Expérience 7 : déconvolution aveugle ===\n');
rng(cfg.seed + 500);
N = cfg.N;
L = cfg.kernel_len;
htrue = make_kernel(cfg.kernel_type, cfg.kernel_param, L);
ops_true = build_ops(htrue, N);
h_wrong = make_kernel('gaussian', 1, L);
ops_wrong = build_ops(h_wrong, N);

% lambda de la déconvolution finale, réglé avec le vrai noyau
% (cadre favorable, à discuter dans le rapport)

[Xva, Yva] = make_dataset(20, N, ops_true, cfg.bsnr, cfg.signal_type);
lam = tune_lambda(@(Y, l) admm_tv(Y, ops_true, l, struct('maxit', 300)), Yva, Xva, logspace(-3, -0.5, 15));

% Un signal

x = gen_signal(cfg.signal_type, N);
y = degrade(x, ops_true, cfg.bsnr);

common = struct('lambda_final', lam, 'xtrue', x, 'htrue', htrue);
[x_l0, h_l0, i_l0] = blind_deconv_am(y, L, setfield(common, 'prior', 'l0')); %#ok<SFLD>

[x_tv, h_tv, i_tv] = blind_deconv_am(y, L, setfield(common, 'prior', 'tv')); %#ok<SFLD>
x_nb = admm_tv(y, ops_true, lam, struct('maxit', 300));
x_wr = admm_tv(y, ops_wrong, lam, struct('maxit', 300));
s1 = [aligned_snr(x, x_nb), aligned_snr(x, x_l0), aligned_snr(x, x_tv), aligned_snr(x, x_wr)];

fprintf('Erreur relative sur le noyau : L0 %.3f | TV %.3f\n', ...
    norm(h_l0 - htrue) / norm(htrue), norm(h_tv - htrue) / norm(htrue));

t = -(L - 1) / 2:(L - 1) / 2;
fig = figure('Name', 'Exp7 - Déconvolution aveugle', 'Position', [100 100 1100 650]);
subplot(2, 2, 1);
plot(t, htrue, 'k', 'LineWidth', 2); hold on;
plot(t, h_l0, 'o-', 'LineWidth', 1.2); plot(t, h_tv, 's-', 'LineWidth', 1.2);
plot(t, i_l0.h0, '--', 'LineWidth', 1.2);
grid on; legend('h vrai', 'h estimé (L0)', 'h estimé (TV)', 'h initial'); title('Noyau'); xlabel('t');
subplot(2, 2, 2);
plot(i_l0.kerr, 'o-', 'LineWidth', 1.3); hold on; plot(i_tv.kerr, 's-', 'LineWidth', 1.3);
grid on; xlabel('itération externe'); ylabel('||h - h_{vrai}|| / ||h_{vrai}||');
legend('a priori L0', 'a priori TV'); title('Erreur sur le noyau');
subplot(2, 1, 2);
plot(x, 'k', 'LineWidth', 2); hold on;
plot(x_nb, 'LineWidth', 1.1); plot(x_l0, 'LineWidth', 1.1); plot(x_tv, 'LineWidth', 1.1);
grid on; xlabel('n');
legend('x', sprintf('non aveugle (%.1f dB)', s1(1)), sprintf('aveugle L0 (%.1f dB)', s1(2)), ...
    sprintf('aveugle TV (%.1f dB)', s1(3)));
title('Reconstructions');
save_figure(fig, cfg, 'exp07_aveugle');

%  Statistiques sur 20 signaux 

nmc = 20;
R = zeros(nmc, 4); K = zeros(nmc, 2);
for m = 1:nmc
    xm = gen_signal(cfg.signal_type, N);
    ym = degrade(xm, ops_true, cfg.bsnr);
    [a, ha] = blind_deconv_am(ym, L, struct('prior', 'l0', 'lambda_final', lam));
    [b, hb] = blind_deconv_am(ym, L, struct('prior', 'tv', 'lambda_final', lam));
    R(m, :) = [aligned_snr(xm, admm_tv(ym, ops_true, lam, struct('maxit', 300))), ...
               aligned_snr(xm, a), aligned_snr(xm, b), ...
               aligned_snr(xm, admm_tv(ym, ops_wrong, lam, struct('maxit', 300)))];
    K(m, :) = [norm(ha - htrue), norm(hb - htrue)] / norm(htrue);
end
names = {'Non aveugle (vrai noyau)'; 'Aveugle, a priori L0'; 'Aveugle, a priori TV'; 'Noyau faux (sigma = 1)'};
T = table(names, mean(R)', std(R)', [NaN; mean(K)'; NaN], ...
    'VariableNames', {'Methode', 'SNR_moyen_dB', 'Ecart_type', 'Erreur_noyau'});
disp(T);
writetable(T, fullfile(cfg.results_dir, 'exp07_aveugle.csv'));
end

function s = aligned_snr(x, xh)
%ALIGNED_SNR Meilleur SNR sur les décalages circulaires de -5 à +5.
s = -Inf;
for d = -5:5
    s = max(s, snr_db(x, circshift(xh, d)));
end
end


%Extension : noyau inconnu.
%   Minimisation alternee (étape x parcimonieuse, étape h par gradient
%   projete sur le simplexe) avec continuation sur lambda, puis ADMM-TV
%   avec le noyau estimé. On compare :
%   - la déconvolution non aveugle avec le vrai noyau (borne supérieure)
%   - l'aveugle avec a priori L0 sur le gradient
%   - l'aveugle avec a priori TV (échec attendu : solution triviale)
%   - la déconvolution avec un noyau faux (gaussien sigma = 1)
%   Le SNR est calcule après recalage (ambiguïté de translation x/h).
%   Résultats sur un signal (figures) puis moyenne sur 20 signaux.