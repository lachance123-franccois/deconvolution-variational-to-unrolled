function M = get_trained_models(cfg)
%Charge (ou entraîne puis sauvegarde) tout ce qui est
%   appris ou réglé sur données, partagé par les expériences 4 à 6 :
%     M.lam_tik, M.lam_tv  : lambdas réglés sur le jeu de validation
%     M.unrolled, M.unrolled_hist : Chambolle-Pock déroulé
%     M.cnn, M.cnn_info    : CNN boîte noire (y -> x)
%     M.denoiser, M.rho_pnp: débruiteur CNN et rho du Plug-and-Play
%   Le cache models/trained_models.mat est réutilisé tant que la
%   configuration ne change pas (sauf cfg.retrain = true).
if nargin < 1, cfg = default_config(); end
f = fullfile(cfg.models_dir, 'trained_models.mat');
sig = model_signature(cfg);
if exist(f, 'file') && ~cfg.retrain
    S = load(f, 'M');
    if isfield(S.M, 'signature') && isequal(S.M.signature, sig)
        M = S.M;
        fprintf('Modèles chargés depuis %s\n', f);
        return;
    end
    fprintf('La configuration a changé : ré-entraînement.\n');
end

rng(cfg.seed + 100);
N = cfg.N;
ops = build_ops(make_kernel(cfg.kernel_type, cfg.kernel_param, cfg.kernel_len), N);
Hm = conv_matrix(ops.Hf);
[Xtr, Ytr] = make_dataset(cfg.n_train, N, ops, cfg.bsnr, cfg.train_types);
[Xva, Yva] = make_dataset(cfg.n_val, N, ops, cfg.bsnr, cfg.train_types);

M.signature = sig;
fprintf('Réglage de lambda (Tikhonov, TV) sur la validation...\n');
M.lam_tik = tune_lambda(@(Y, l) tikhonov_deconv(Y, ops, l, 'gradient'), ...
    Yva, Xva, logspace(-5, 1, 40));
M.lam_tv = tune_lambda(@(Y, l) admm_tv(Y, ops, l, struct('maxit', 300)), ...
    Yva, Xva, logspace(-3, -0.5, 20));

M.has_dl = has_dl_toolbox();
if ~M.has_dl
    warning('Deep Learning Toolbox absente : seules les méthodes classiques seront disponibles.');
else
    fprintf('Entraînement du Chambolle-Pock déroulé...\n');
    [M.unrolled, M.unrolled_hist] = train_unrolled_cp(Xtr, Ytr, Hm, Xva, Yva, ...
        struct('layers', cfg.unrolled_layers, 'epochs', cfg.unrolled_epochs, ...
               'batch', cfg.unrolled_batch, 'lr', cfg.unrolled_lr, 'lam0', cfg.unrolled_lam0));

    cnn_opts = struct('depth', cfg.cnn_depth, 'filters', cfg.cnn_filters, ...
                      'fsize', cfg.cnn_fsize, 'epochs', cfg.cnn_epochs);
    fprintf('Entraînement du CNN boîte noire (y -> x)...\n');
    [M.cnn, M.cnn_info] = train_cnn1d(Ytr, Xtr, Yva, Xva, cnn_opts);

    fprintf('Entraînement du débruiteur CNN (Plug-and-Play)...\n');
    sd_tr = cfg.denoiser_sigma_max * rand(1, size(Xtr, 2));
    sd_va = cfg.denoiser_sigma_max * rand(1, size(Xva, 2));
    [M.denoiser, M.denoiser_info] = train_cnn1d(Xtr + sd_tr .* randn(size(Xtr)), Xtr, ...
        Xva + sd_va .* randn(size(Xva)), Xva, cnn_opts);

    fprintf('Réglage de rho (PnP-ADMM)...\n');
    den = M.denoiser;
    M.rho_pnp = tune_lambda(@(Y, r) pnp_admm(Y, ops, @(v, ~) cnn_apply(den, v), ...
        struct('rho', r, 'maxit', 50)), Yva, Xva, logspace(-2, 1, 10));
end
save(f, 'M');
fprintf('Modèles sauvegardés dans %s\n', f);
end

function s = model_signature(cfg)
s = struct('N', cfg.N, 'kernel_type', cfg.kernel_type, ...
    'kernel_param', cfg.kernel_param, 'kernel_len', cfg.kernel_len, ...
    'bsnr', cfg.bsnr, 'n_train', cfg.n_train, 'train_types', {cfg.train_types}, ...
    'layers', cfg.unrolled_layers, 'cnn_depth', cfg.cnn_depth);
end
