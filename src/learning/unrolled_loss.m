function [loss, grad] = unrolled_loss(params, Y, X, Hm, S)
%Erreur quadratique moyenne et gradient (à appeler via dlfeval).
Xhat = unrolled_cp_forward(params, Y, Hm, S);
loss = mean((Xhat - X).^2, 'all');
grad = dlgradient(loss, params);
end
