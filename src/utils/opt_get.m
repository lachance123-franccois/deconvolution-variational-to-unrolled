function v = opt_get(opts, name, default)
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end
