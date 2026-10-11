function [lambda, info] = select_lambda(y, ops, method, opts)

if nargin < 4, opts = struct(); end
reg = opt_get(opts, 'reg', 'gradient');
lambdas = opt_get(opts, 'lambdas', logspace(-6, 2, 200));
lambdas = lambdas(:);
N = ops.N;
Y = fft(y);
H2 = abs(ops.Hf).^2;
R2 = reg_spectrum(ops, reg);

nl = numel(lambdas);
res = zeros(nl, 1); semi = zeros(nl, 1); trc = zeros(nl, 1);
for i = 1:nl
    den = H2 + lambdas(i) * R2;
    phi = H2 ./ den;
    X = conj(ops.Hf) .* Y ./ den;
    res(i)  = sum(abs((1 - phi) .* Y).^2) / N;   % ||y - Hx||^2 

    semi(i) = sum(R2 .* abs(X).^2) / N;          % ||Rx||^2
    trc(i)  = sum(1 - phi);                      % trace(I - A(lambda))
end
info.lambdas = lambdas;
info.residual = res;
info.seminorm = semi;

switch lower(method)
    case 'gcv'
        G = N * res ./ trc.^2;
        [~, k] = min(G);
        info.curve = G;
    case 'lcurve'
        t = log(lambdas);
        rho = log(res); eta = log(semi);
        dr = gradient(rho, t);  
        de = gradient(eta, t);
        ddr = gradient(dr, t); 
        dde = gradient(de, t);
        kappa = (dr .* dde - ddr .* de) ./ (dr.^2 + de.^2).^(3/2);
        kappa([1:2, end-1:end]) = -Inf;          % ignore les bords
        [~, k] = max(kappa);
        info.curve = kappa;
    case 'discrepancy'
        sigma = opt_get(opts, 'sigma', []);
        if isempty(sigma)
            error('select_lambda:sigma', 'Morozov nécessite opts.sigma.');
        end
        target = opt_get(opts, 'tau', 1) * N * sigma^2;
        [~, k] = min(abs(log(res) - log(target)));
        info.curve = res;
        info.target = target;
    otherwise
        error('select_lambda:method', 'Méthode inconnue : %s', method);
end
lambda = lambdas(k);
info.index = k;
end

%SELECT_LAMBDA Choix automatique du paramètre de Tikhonov.
%   method : 'gcv'         validation croisée généralisée
%            'lcurve'      coin de la courbe en L (courbure maximale)
%            'discrepancy' principe de discordance de Morozov
%                          (||y - Hx||^2 = tau N sigma^2), exige opts.sigma
%   opts.reg     : régularisation ('gradient' par défaut)
%   opts.lambdas : grille de recherche (logspace(-6, 2, 200) par défaut)
%   opts.tau     : facteur de sécurité de Morozov (1 par défaut)
%
%   Tout est calculé dans le domaine de Fourier grâce aux facteurs de
%   filtrage phi(f) = |H|^2 / (|H|^2 + lambda |R|^2).
