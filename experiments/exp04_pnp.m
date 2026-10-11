function T = exp04_pnp(cfg)
    if nargin < 1, cfg = default_config(); end
    fprintf('\n=== Expérience 4 : Plug-and-Play ADMM ===\n');
    M = get_trained_models(cfg);
    N = cfg.N;
    ops = build_ops(make_kernel(cfg.kernel_type, cfg.kernel_param, cfg.kernel_len), N);
    rng(cfg.seed + 300);
    [Xte, Yte] = make_dataset(50, N, ops, cfg.bsnr, cfg.train_types);
    
    den_tv = @(v, rho) tv_denoise(v, M.lam_tv / rho, 100);
    names = {'ADMM-TV'; 'PnP-ADMM (débruiteur TV)'};
    X1 = admm_tv(Yte, ops, M.lam_tv, struct('maxit', 300));
    X2 = pnp_admm(Yte, ops, den_tv, struct('rho', 0.1, 'maxit', 100));
    snrs = {snr_db(Xte, X1), snr_db(Xte, X2)};
    
    if M.has_dl
        den_cnn = @(v, ~) cnn_apply(M.denoiser, v);
        X3 = pnp_admm(Yte, ops, den_cnn, struct('rho', M.rho_pnp, 'maxit', 50));
        names{end + 1} = 'PnP-ADMM (débruiteur CNN)';
        snrs{end + 1} = snr_db(Xte, X3);
    
        % sensibilité à rho
        rhos = logspace(-2, 1, 10);
        sr = zeros(size(rhos));
        for i = 1:numel(rhos)
            sr(i) = mean(snr_db(Xte, pnp_admm(Yte, ops, den_cnn, struct('rho', rhos(i), 'maxit', 50))));
        end
        fig = figure('Name', 'Exp4 - Sensibilité à rho', 'Position', [100 100 600 380]);
        semilogx(rhos, sr, 'o-', 'LineWidth', 1.4); grid on;
        xlabel('\rho'); ylabel('SNR moyen (dB)');
        title('PnP-ADMM (CNN) : sensibilité à \rho');
        save_figure(fig, cfg, 'exp04_sensibilite_rho');
    
        % convergence sur un signal
        x = Xte(:, 1); y = Yte(:, 1);
        [~, iT] = pnp_admm(y, ops, den_tv, struct('rho', 0.1, 'maxit', 100, 'tol', 0, 'xtrue', x));
        [~, iN] = pnp_admm(y, ops, den_cnn, struct('rho', M.rho_pnp, 'maxit', 100, 'tol', 0, 'xtrue', x));
        [~, iA] = admm_tv(y, ops, M.lam_tv, struct('maxit', 100, 'tol', 0, 'xtrue', x));
        fig = figure('Name', 'Exp4 - Convergence', 'Position', [100 100 650 400]);
        plot(iA.snr, 'LineWidth', 1.3); 
        hold on; plot(iT.snr, 'LineWidth', 1.3);
        plot(iN.snr, 'LineWidth', 1.3);
        grid on; 
        xlabel('itération'); 
        ylabel('SNR (dB)');
        legend('ADMM-TV', 'PnP (TV)', 'PnP (CNN)', 'Location', 'southeast');
        title('Plug-and-Play : SNR au fil des itérations');
        save_figure(fig, cfg, 'exp04_convergence');
    
        fig = figure('Name', 'Exp4 - Exemple', 'Position', [100 100 900 420]);
        plot(Xte(:, 2), 'k', 'LineWidth', 2);
        hold on;
        plot(X1(:, 2), 'LineWidth', 1.2); 
        plot(X3(:, 2), 'LineWidth', 1.2);
        legend('x', sprintf('ADMM-TV (%.1f dB)', snrs{1}(2)), sprintf('PnP CNN (%.1f dB)', snrs{3}(2)));
        grid on; xlabel('n'); title('Exemple de reconstruction');
        save_figure(fig, cfg, 'exp04_exemple');
    end
    
    m = cellfun(@mean, snrs)'; s = cellfun(@std, snrs)';
    T = table(names, m, s, 'VariableNames', {'Methode', 'SNR_moyen_dB', 'Ecart_type'});
    disp(T);
    writetable(T, fullfile(cfg.results_dir, 'exp04_pnp.csv'));
end

%EXP04_PNP Plug-and-Play ADMM.
%   - PnP avec débruiteur TV : contrôle de cohérence (même minimiseur
%     que l'ADMM-TV, par un découpage différent)
%   - PnP avec débruiteur CNN appris (Deep Learning Toolbox)
%   - sensibilité à rho et convergence


