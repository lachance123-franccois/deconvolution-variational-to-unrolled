function v = opt_get(opts, name, default)
    %Ceci lit opts.(name) s'il existe et n'est pas vide, sinon renvoie default.
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        v = opts.(name);
    else
        v = default;
    end
end
