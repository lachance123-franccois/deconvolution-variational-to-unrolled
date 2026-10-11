function ok = run_tests()
%RUN_TESTS Tests unitaires : adjoints, convolution, ondelettes, solveurs.
%   Renvoie true si tous les tests passent.
fprintf('=== Tests unitaires ===\n');
rng(1);
nfail = 0;
N = 64;

% Opérateurs 
h = make_kernel('gaussian', 2, 11);
ops = build_ops(h, N);
x = randn(N, 1); y = randn(N, 1);
nfail = nfail + check('Noyau de somme 1', abs(sum(h) - 1) < 1e-12);
nfail = nfail + check('Adjoint de H', abs(ops.H(x)' * y - x' * ops.Ht(y)) < 1e-10);
nfail = nfail + check('Adjoint de D', abs(ops.D(x)' * y - x' * ops.Dt(y)) < 1e-10);

t = -(numel(h) - 1) / 2:(numel(h) - 1) / 2;
hx = zeros(N, 1);
for j = 1:numel(h), hx = hx + h(j) * circshift(x, t(j)); end
nfail = nfail + check('Convolution circulaire centrée', norm(hx - ops.H(x)) < 1e-10);
nfail = nfail + check('D = différence avant', norm(ops.D(x) - (circshift(x, -1) - x)) < 1e-10);
nfail = nfail + check('Matrice de convolution', norm(conv_matrix(ops.Hf) * x - ops.H(x)) < 1e-10);

S = shift_matrices(N);
Dm = 0 * S{1} - S{2} + S{3};
nfail = nfail + check('Matrices de décalage', norm(Dm * x - ops.D(x)) < 1e-10);

% Ondelettes 
c = haar_dwt(x, 3);
nfail = nfail + check('Haar : reconstruction parfaite', norm(haar_idwt(c, 3) - x) < 1e-10);
nfail = nfail + check('Haar : orthonormalité', abs(norm(c) - norm(x)) < 1e-10);

% Tikhonov : lambda -> 0 sans bruit = inverse exacte 
ops_e = build_ops(make_kernel('exponential', 1, 11), N);   % |H(f)| > 0
xt = tikhonov_deconv(ops_e.H(x), ops_e, 1e-12, 'identity');
nfail = nfail + check('Tikhonov, lambda -> 0', snr_db(x, xt) > 60);

% TV : ADMM et Chambolle-Pock atteignent le même minimum 
xpc = gen_signal('piecewise', N);
y2 = ops.H(xpc) + 0.01 * randn(N, 1);
lam = 0.02;
[~, ia] = admm_tv(y2, ops, lam, struct('maxit', 5000, 'tol', 1e-10));
[~, ic] = chambolle_pock_tv(y2, ops, lam, struct('maxit', 20000, 'tol', 1e-12));
Fa = ia.obj(end); Fc = ic.obj(end);
nfail = nfail + check('ADMM-TV et CP-TV : même objectif', abs(Fa - Fc) / Fa < 1e-3);

% tv_denoise = ADMM-TV avec H = identité 
v = xpc + 0.1 * randn(N, 1);
ops_id = build_ops(make_kernel('delta', 0, 1), N);
x1 = tv_denoise(v, 0.1, 5000);
x2 = admm_tv(v, ops_id, 0.1, struct('maxit', 5000, 'tol', 1e-10));
nfail = nfail + check('Débruitage TV (dual) = ADMM', norm(x1 - x2) / norm(x2) < 1e-3);

%  Projection sur le simplexe 
w = proj_simplex(randn(7, 1));
nfail = nfail + check('Projection simplexe', abs(sum(w) - 1) < 1e-12 && all(w >= 0));

% Choix de lambda : s'exécute et renvoie une valeur de la grille 
[y3, s3] = degrade(xpc, ops, 30);
crit = {'gcv', 'lcurve', 'discrepancy'};
for i = 1:numel(crit)
    l = select_lambda(y3, ops, crit{i}, struct('sigma', s3));
    nfail = nfail + check(['select_lambda ' crit{i}], isscalar(l) && l > 0);
end

ok = (nfail == 0);
if ok
    fprintf('Tous les tests sont passés.\n\n');
else
    warning('%d test(s) en échec.', nfail);
end
end

function f = check(name, cond)
if cond
    fprintf('  [OK]    %s\n', name);
    f = 0;
else
    fprintf('  [ECHEC] %s\n', name);
    f = 1;
end
end
