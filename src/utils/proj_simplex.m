function w = proj_simplex(v)
    %PROJ_SIMPLEX fait la Projection euclidienne sur le simplexe {w >= 0, sum(w) = 1}.
    v = v(:);
    n = numel(v);
    u = sort(v, 'descend');
    css = cumsum(u);
    r = find(u - (css - 1) ./ (1:n)' > 0, 1, 'last');
    theta = (css(r) - 1) / r;
    w = max(v - theta, 0);
end
