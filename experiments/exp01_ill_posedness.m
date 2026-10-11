function exp01_ill_posedness(cfg)
%Mise en evidence du caractere mal pose du probleme de deconvolution  par la methode directe.
%   1) Spectre |H(f)| (= valeurs singulières de H circulante) et
%      conditionnement pour plusieurs largeurs de flou.
%   2) Inversion naïve : parfaite sans  le bruit, catastrophique avec le bruit.
%   3) Diagramme de Picard : |Y(f)| vs |H(f)|.
%   4) SNR de l'inversion naïve en fonction du BSNR.
if nargin < 1, cfg = default_config(); end
fprintf('\n Expérience 1 :probleme  mal pose =\n');
rng(cfg.seed);
N = cfg.N;

% 1) Spectres et conditionnement 

widths = [1 2 3 5];
f = (-N/2:N/2 - 1)' / N;
fig = figure('Name', 'Exp1 - Spectre du noyau', 'Position', [100 100 700 420]);
leg = cell(size(widths));
for i = 1:numel(widths)
    Hf = kernel_fft(make_kernel('gaussian', widths(i), cfg.kernel_len), N);
    semilogy(f, fftshift(abs(Hf)) + eps, 'LineWidth', 1.4); hold on;
    leg{i} = sprintf('\\sigma_h = %g  (cond(H) = %.1e)', widths(i), max(abs(Hf)) / min(abs(Hf)));
end
grid on;
xlabel('Fréquence normalisee'); 
ylabel('|H(f)|');
title('Fonction de transfert du flou gaussien = valeurs singulières de H');
legend(leg, 'Location', 'south');
save_figure(fig, cfg, 'exp01_spectre_noyau');

% 2) Inversion naïve 
ops = build_ops(make_kernel(cfg.kernel_type, cfg.kernel_param, cfg.kernel_len), N);
x = gen_signal(cfg.signal_type, N);
[y, sigma, Hx] = degrade(x, ops, cfg.bsnr);
x_clean = naive_inverse(Hx, ops);
x_noisy = naive_inverse(y, ops);
fprintf('Inversion naïve sans bruit : SNR = %6.1f dB\n', snr_db(x, x_clean));
fprintf('Inversion naïve avec bruit (BSNR %d dB) : SNR = %6.1f dB\n', cfg.bsnr, snr_db(x, x_noisy));

fig = figure('Name', 'Exp1 - Inversion naïve', 'Position', [100 100 900 650]);
subplot(3, 1, 1);
plot(x, 'k', 'LineWidth', 1.5);
hold on; 
plot(y, 'Color', [0.85 0.33 0.1]);
legend('x (vrai)', 'y = Hx + n'); 
title('Signal et observation'); grid on;
subplot(3, 1, 2);
plot(x, 'k', 'LineWidth', 1.5);
hold on; 
plot(x_clean, '--');
title(sprintf('Inversion naïve sans bruit : SNR = %.1f dB', snr_db(x, x_clean))); grid on;
subplot(3, 1, 3);
plot(x_noisy);
title(sprintf('Inversion naïve avec bruit : SNR = %.1f dB', snr_db(x, x_noisy))); grid on;
xlabel('n');
save_figure(fig, cfg, 'exp01_inversion_naive');

% 3) Diagramme de Picard 
[s, idx] = sort(abs(ops.Hf), 'descend');
Y = fft(y);
fig = figure('Name', 'Exp1 - Picard', 'Position', [100 100 700 420]);
semilogy(s, 'LineWidth', 1.5); hold on;
semilogy(abs(Y(idx)) / sqrt(N), '.');
semilogy(abs(Y(idx)) ./ s / sqrt(N), '.');
yline(sigma, '--k', '\sigma (niveau de bruit)');
grid on; xlabel('indice (fréquences triées par |H| décroissant)');
legend('|H(f)|', '|Y(f)|/\surd N', '|Y(f)|/|H(f)|/\surd N', 'Location', 'southwest');
title('Diagramme de Picard : |Y| atteint le plancher de bruit, |Y|/|H| explose');
save_figure(fig, cfg, 'exp01_picard');

% 4) SNR vs BSNR 
bsnrs = 20:10:100;
snrs = zeros(size(bsnrs));
for i = 1:numel(bsnrs)
    yi = degrade(x, ops, bsnrs(i));
    snrs(i) = snr_db(x, naive_inverse(yi, ops));
end
fig = figure('Name', 'Exp1 - SNR vs BSNR', 'Position', [100 100 600 380]);
plot(bsnrs, snrs, 'o-', 'LineWidth', 1.4); grid on;
xlabel('BSNR (dB)'); 
ylabel('SNR de reconstruction (dB)');
title('Inversion naïve : il faut un bruit infime pour un résultat utilisable');
save_figure(fig, cfg, 'exp01_snr_vs_bsnr');
end
