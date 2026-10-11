function lgraph = build_cnn1d(N, depth, nf, fsize)
%BUILD_CNN1D CNN 1D résiduel de type DnCNN (Zhang et al., 2017).
%   Les signaux sont traités comme des images N x 1 x 1.
%   sortie = entrée + correction apprise (apprentissage résiduel)
%   depth : nombre de couches de convolution ; nf : nombre de filtres ;
%   fsize : taille des filtres. Champ réceptif = depth*(fsize-1) + 1.
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
