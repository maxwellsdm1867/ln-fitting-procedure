function b = cascadeBic(resp, pred, nFreeParams)
%CASCADEBIC  BIC for a Gaussian-residual fit -- the criterion for model comparison.
% R^2 alone always favours the bigger model, because the richer models nest the simpler ones.
n = numel(resp);
b = n * log(sum((resp(:) - pred(:)).^2) / n) + nFreeParams * log(n);
end
