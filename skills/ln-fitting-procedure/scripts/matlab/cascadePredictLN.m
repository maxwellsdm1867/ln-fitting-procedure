function pred = cascadePredictLN(params, stim, dt)
%CASCADEPREDICTLN  LN prediction using Fred's nodes, so there is one model definition.
filt = ParamFilterNode.getFilterWithParams(params, size(stim,2), dt);
g    = real(ifft(fft(stim') .* fft(filt)))';
pred = SigmoidNlNode.processTempParams([params.alpha; params.beta; params.gamma; params.epsilon], g);
end
