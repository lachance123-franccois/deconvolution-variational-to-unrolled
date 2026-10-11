%% MAIN - Déconvolution de signaux : des méthodes variationnelles aux réseaux déroulés
%
%  Lance l'ensemble du projet : tests unitaires puis les 7 expériences.
%  Chaque expérience peut aussi être lancée seule (voir README.md).
%  Les figures (.png) et tableaux (.csv) sont écrits dans results/.

clear; close all; clc;
startup_project();
cfg = default_config();

run_tests();
%%
exp02_linear_methods(cfg);     % Wiener, Tikhonov, choix de lambda
%%
exp03_tv_methods(cfg);         % TV : ADMM, Chambolle-Pock ; FISTA-ondelettes
exp04_pnp(cfg);                % Plug-and-Play ADMM (TV, CNN)
exp05_learned_methods(cfg);    % réseau déroulé vs CNN boîte noire
exp06_robustness(cfg);         % bruit, erreur de noyau, changement de noyau
exp07_blind_deconvolution(cfg);% extension : déconvolution aveugle

fprintf('\nTerminé. Résultats dans : %s\n', cfg.results_dir);
