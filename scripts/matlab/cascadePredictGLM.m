function pred = cascadePredictGLM(p, stim, dt)
%CASCADEPREDICTGLM  Free-running GLM prediction from a parameter struct.
f = cascadeFilterSafe([p.numFilt p.tauR p.tauD p.tauP p.phi], size(stim,2), dt);
if isempty(f), pred = []; return; end
nFb = 30; if isfield(p,'n_fb_bins'), nFb = p.n_fb_bins; end
pred = cascadeFreeRun(cascadeConv(stim, f), p.alpha, p.beta, p.gamma, p.epsilon, ...
                      p.a_fb, p.tau_fb, nFb, dt);
end
