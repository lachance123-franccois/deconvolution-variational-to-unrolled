function n = count_learnables(net)
n = 0;
props = {'Weights', 'Bias', 'Offset', 'Scale'};
for i = 1:numel(net.Layers)
    L = net.Layers(i);
    for j = 1:numel(props)
        if isprop(L, props{j})
            n = n + numel(L.(props{j}));
        end
    end
end
end
