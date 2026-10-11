function R2 = reg_spectrum(ops, reg)
switch lower(reg)
    case 'identity'
        R2 = ones(ops.N, 1);
    case 'gradient'
        R2 = abs(ops.Df).^2;
    case 'laplacian'
        R2 = abs(ops.Df).^4;
    otherwise
        error('reg_spectrum:reg', 'Régularisation inconnue : %s', reg);
end
end
%REG_SPECTRUM |R(f)|^2 pour l'opérateur de régularisation de Tikhonov.
%   reg : 'identity' (R = I), 'gradient' (R = D), 'laplacian' (R = D^T D)