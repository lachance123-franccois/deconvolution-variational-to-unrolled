function h = make_kernel(type, param, L)
%Réponse impulsionnelle centrée, de longueur impaire L, de somme 1.
%   type : 'gaussian'    param = écart-type (échantillons)
%          'uniform'     param = largeur (moyenne glissante)
%          'exponential' param = constante de temps (noyau causal, asymétrique)
%          'delta'       identité (pas de flou)
if nargin < 3 || isempty(L), L = 2 * ceil(3 * param) + 1; end
if mod(L, 2) == 0, L = L + 1; end 
t = (-(L - 1) / 2:(L - 1) / 2)';
switch lower(type) % ceci nous permet d'eviter la casse, cad d'etre sensible a la majuscule dans l'ecriture du type. "gaussian", "Gaussian", "gAussian", focntionne de la meme maniere 
    case 'gaussian'
        h = exp(-t.^2 / (2 * param^2));
    case 'uniform'
        h = double(abs(t) < param / 2);
    case 'exponential'
        h = exp(-t / param) .* (t >= 0);
    case 'delta'
        h = double(t == 0);
    otherwise
        error('make_kernel:type', 'Type de noyau inconnu : %s', type);
end
h = h / sum(h);
end
