function F = tv_objective(x, y, ops, lambda)
%TV_OBJECTIVE  F(x) = 1/2 ||Hx - y||^2 + lambda ||Dx||_1 (somme sur les colonnes)
F = 0.5 * sum((ops.H(x) - y).^2, 'all') + lambda * sum(abs(ops.D(x)), 'all');
end
