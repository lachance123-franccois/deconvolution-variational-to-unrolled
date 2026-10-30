function [loss, grad] = unrolled_loss(params, Y, X, Hm, S)
Xhat = unrolled_cp_forward(params, Y, Hm, S);
loss = mean((Xhat - X).^2, 'all');
grad = dlgradient(loss, params);
end
