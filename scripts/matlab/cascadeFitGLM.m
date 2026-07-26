function out = cascadeFitGLM(stim, resp, dt, varargin)
%CASCADEFITGLM  Staged fit of the LN model plus an exponential feedback kernel (11 params).
%
%   out = cascadeFitGLM(stim, resp, dt)
%   out = cascadeFitGLM(stim, resp, dt, 'nFbBins', 30, 'nRandomInits', 20, 'seed', 0)
%
% Free-running throughout: the feedback is driven by the model's own predictions, never by the
% observed response. See cascadeFreeRun for why teacher forcing invalidates the fit.
%
% Interpreting the result: the GLM nests the LN at a_fb = 0, so the bar is that it does as
% well as the LN, NOT that it beats it. A GLM that converges with feedback effectively off is
% a correct result on a cell without feedback. A GLM materially BELOW the LN means its own
% optimization failed, since the LN is inside its search space.
%
% out.params includes a_fb and tau_fb. Judge the feedback by the DIMENSIONLESS loop gain
% out.loopGain = a_fb * sum_k exp(-k*dt/tau_fb) * alpha * beta * phi(0), never by a_fb alone:
% the slope term is signed, so a negative a_fb against a negative alpha is REGENERATIVE.

p = inputParser;
p.addParameter('nFbBins', 30, @isscalar);
p.addParameter('nRandomInits', 20, @isscalar);
p.addParameter('nRestarts', 10, @isscalar);
p.addParameter('seed', 0, @isscalar);
p.addParameter('diagnose', true, @islogical);
p.addParameter('verbose', true, @islogical);
p.parse(varargin{:});
opt = p.Results;
assert(exist('ParamFilterNode','class') == 8, 'cascadeFitGLM:noCascadeGraph', ...
    'CascadeGraph is not on the path.');

rng(opt.seed);
PENALTY = 1e12; nPts = size(stim,2); nFb = opt.nFbBins;

% ---- stage 1: filter only (feedback cannot help find the filter, and widens the search)
[pFilt, losses, ef1] = cascadeStage1Filter(stim, resp, dt, opt.nRandomInits, opt.nRestarts);
f = cascadeFilterSafe(pFilt, nPts, dt);
x = cascadeConv(stim, f);

% ---- stage 2: NL + feedback jointly, free-running, several feedback starts
qNL = cascadeStage2NL(x, resp, opt.nRestarts);
rScale = max(resp(:)) - min(resp(:));
scale  = std(x(:)) / max(rScale, eps);      % put the amplitudes on the drive's scale
fbInits = [0, 5*dt; -0.01*rScale*scale, dt; -0.05*rScale*scale, 5*dt; ...
           0.05*rScale*scale, 5*dt; -0.01*rScale*scale, 20*dt];

loss2 = @(v) glmLoss2(v, x, resp, nFb, dt, PENALTY);
best2 = []; bestF2 = inf;
for i = 1:size(fbInits,1)
    [v, fv] = cascadeNmRestarts(loss2, [qNL(:)' fbInits(i,:)], opt.nRestarts);
    if fv < bestF2, best2 = v; bestF2 = fv; end
end

% ---- stage 3: joint, all 11, free-running
loss3 = @(v) glmLoss3(v, stim, resp, dt, nPts, nFb, PENALTY);
[vJ, ~, ef3] = cascadeNmRestarts(loss3, [pFilt(:)' best2(:)'], opt.nRestarts);

params = struct('numFilt',vJ(1),'tauR',vJ(2),'tauD',vJ(3),'tauP',vJ(4),'phi',vJ(5), ...
                'alpha',vJ(6),'beta',vJ(7),'gamma',vJ(8),'epsilon',vJ(9), ...
                'a_fb',vJ(10),'tau_fb',max(vJ(11), dt),'n_fb_bins',nFb);
params = cascadeCanonical(params, dt);
pred = cascadePredictGLM(params, stim, dt);
r2 = computeVarianceExplained(pred, resp);

decay = sum(exp(-(1:nFb)' * dt / max(params.tau_fb, dt)));
loopGain = params.a_fb * decay * params.alpha * params.beta * (1/sqrt(2*pi));

out = struct('params', params, 'r2PerEpoch', r2(:)', 'r2Mean', mean(r2), 'dt', dt, ...
             'loopGain', loopGain, ...
             'feedbackType', ternary(loopGain < 0, 'adaptive', 'regenerative'));
if opt.diagnose
    ef = ef3; if isempty(ef), ef = ef1; end
    out.diagnostics = cascadeDiagnose(params, stim, resp, dt, losses, ef, r2, loss3, 'glm');
    if opt.verbose && ~out.diagnostics.ok
        for k = 1:numel(out.diagnostics.warnings)
            fprintf('[cascadeFit] WARNING: %s\n', out.diagnostics.warnings{k});
        end
    end
end
if opt.verbose
    fprintf('[cascadeFit] GLM loop gain %+.3f (%s); |gain|>1 means the recursion is held only by saturation\n', ...
            loopGain, out.feedbackType);
end
end

function e = glmLoss2(v, x, resp, nFb, dt, PENALTY)
pred = cascadeFreeRun(x, v(1), v(2), v(3), v(4), v(5), max(v(6), dt), nFb, dt);
e = sum((resp(:) - pred(:)).^2);
if ~isfinite(e), e = PENALTY; end
end

function e = glmLoss3(v, stim, resp, dt, nPts, nFb, PENALTY)
f = cascadeFilterSafe(v(1:5), nPts, dt);
if isempty(f), e = PENALTY; return; end
pred = cascadeFreeRun(cascadeConv(stim,f), v(6), v(7), v(8), v(9), v(10), max(v(11), dt), nFb, dt);
e = sum((resp(:) - pred(:)).^2);
if ~isfinite(e), e = PENALTY; end
end

function y = ternary(c, a, b)
if c, y = a; else, y = b; end
end
