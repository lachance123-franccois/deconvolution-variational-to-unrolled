function X = cnn_apply(net, V)
%CNN_APPLY Applique un CNN 1D (train_cnn1d) à des signaux N x B.
[N, B] = size(V);
out = predict(net, reshape(V, N, 1, 1, B));
X = reshape(double(out), N, B);
end
