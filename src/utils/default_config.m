function cfg = default_config()

cfg.root        = fileparts(fileparts(fileparts(mfilename('fullpath'))));
cfg.results_dir = fullfile(cfg.root, 'results');
cfg.models_dir  = fullfile(cfg.root, 'models');
cfg.save_figures = true;
cfg.seed = 0;

% Signal et modèle direct 
cfg.N            = 256;           % longueur des signaux
cfg.signal_type  = 'piecewise_smooth';   % 'piecewise' | 'piecewise_smooth' | 'audio'
cfg.kernel_type  = 'gaussian';    % voir make_kernel pour comprendre, nous avaons aussi ('exponential",'uniform')
cfg.kernel_param = 3;             % écart-type du flou gaussien (échantillons)
cfg.kernel_len   = 31;            % c'est la taille du noyau L  utilise make_kernel
cfg.bsnr         = 30;            % Blurred SNR en dB

% Apprentissage (expériences 4 à 6) 
cfg.retrain      = false;         % true : force le ré-entraînement
cfg.train_types  = {'piecewise', 'piecewise_smooth'};
cfg.n_train      = 4000;
cfg.n_val        = 200;
cfg.n_test       = 200;

% Reseau deroule (Chambolle-Pock appris)
cfg.unrolled_layers = 15;
cfg.unrolled_epochs = 40;
cfg.unrolled_batch  = 32;
cfg.unrolled_lr     = 1e-2;
cfg.unrolled_lam0   = 0.03;       % initialisation du seuil TV

% CNN 1D (boîte noire et débruiteur Plug-and-Play)
cfg.cnn_depth   = 12;
cfg.cnn_filters = 32;
cfg.cnn_fsize   = 5;
cfg.cnn_epochs  = 30;
cfg.denoiser_sigma_max = 0.1;     % niveaux de bruit vus par le débruiteur
end
