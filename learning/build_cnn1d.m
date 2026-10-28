function lgraph = build_cnn1d(N, depth, nf, fsize)
layers = [
    imageInputLayer([N 1 1], 'Normalization', 'none', 'Name', 'in')
    convolution2dLayer([fsize 1], nf, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')];
for d = 2:depth - 1
    layers = [layers
        convolution2dLayer([fsize 1], nf, 'Padding', 'same', 'Name', sprintf('conv%d', d))
        batchNormalizationLayer('Name', sprintf('bn%d', d))
        reluLayer('Name', sprintf('relu%d', d))]; %#ok<AGROW>
end
layers = [layers
    convolution2dLayer([fsize 1], 1, 'Padding', 'same', 'Name', 'conv_out')];
lgraph = layerGraph(layers);
lgraph = addLayers(lgraph, [
    additionLayer(2, 'Name', 'add')
    regressionLayer('Name', 'out')]);
lgraph = connectLayers(lgraph, 'conv_out', 'add/in1');
lgraph = connectLayers(lgraph, 'in', 'add/in2');
end
