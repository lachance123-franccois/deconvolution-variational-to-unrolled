function y = soft_threshold(v, t)
y = sign(v) .* max(abs(v) - t, 0);
end
