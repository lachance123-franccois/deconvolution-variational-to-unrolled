function params = unrolled_cp_init(K, lam0)
%Parametres initiaux du Chambolle-Pock déroulé à K couches.
%   Chaque couche k possede :
%     logtau(k), logsig(k) : pas primal et dual (paramétrés en log => > 0)
%     loglam(k)            : seuil TV (log)
%     W(:,k)               : filtre d'analyse à 3 coefficients (décalages -1,0,+1)
%   Initialisation = l'algorithme classique : W = [0 -1 1] (gradient) et
%   tau = 2, sigma = 0.09 (tau*sigma*||K||^2 < 1 avec ||K||^2 <= 1 + 4).
if nargin < 2, lam0 = 0.03; end
params.logtau = log(2) * ones(1, K);
params.logsig = log(0.09) * ones(1, K);
params.loglam = log(lam0) * ones(1, K);
params.W = repmat([0; -1; 1], 1, K);
end
