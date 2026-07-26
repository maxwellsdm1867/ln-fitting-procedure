function out = cascadeFitLN(stim, resp, dt, varargin)
%CASCADEFITLN  Staged Nelder-Mead fit of the CascadeGraph LN model, with diagnostics.
%
%   out = cascadeFitLN(stim, resp, dt)
%   out = cascadeFitLN(stim, resp, dt, 'nRandomInits', 20, 'nRestarts', 10, 'seed', 0)
%
%   out.params        struct with numFilt tauR tauD tauP phi alpha beta gamma epsilon
%   out.r2PerEpoch    row-wise R^2, one per epoch (CascadeGraph computeVarianceExplained)
%   out.r2Mean        their mean
%   out.diagnostics   .ok, .warnings (cellstr), .checks -- see below
%
% The model itself is Fred's: this calls ParamFilterNode.getFilterWithParams and
% SigmoidNlNode.processTempParams directly, so there is exactly one definition of the filter
% and the nonlinearity and parity with the Python is automatic rather than maintained.
%
% What this adds on top of the nodes is the part CascadeGraph does not provide: the staged
% pipeline, random restarts, the binned-nonlinearity prefit, a finite-loss guard, resolution
% of the four exact degeneracies, and a diagnostics struct so a fit reports its own verdict.
%
% DIAGNOSTICS checks, all automatic:
%   converged          did fminsearch settle, or exhaust MaxFunEvals (exitflag 0)
%   startAgreement     how many of the random starts reached the best loss
%   localOptimum       does perturbing each parameter raise the loss
%   roundtrip          do the returned parameters reproduce the reported R^2
%   causalityLagBins   impulse response peak lag; must be positive
%   filterVsXcorr      |corr| with the cross-correlation filter estimate
%   perEpochR2Spread   max minus min across epochs
%
% Requires CascadeGraph on the path (addpath(genpath(<cascadegraph>))).

p = inputParser;
p.addParameter('nRandomInits', 20, @isscalar);
p.addParameter('nRestarts',    10, @isscalar);
p.addParameter('seed',          0, @isscalar);
p.addParameter('diagnose',   true, @islogical);
p.addParameter('verbose',    true, @islogical);
p.parse(varargin{:});
opt = p.Results;

assert(exist('ParamFilterNode','class') == 8, 'cascadeFitLN:noCascadeGraph', ...
    'CascadeGraph is not on the path; run addpath(genpath(<cascadegraph root>)) first.');

rng(opt.seed);
nPts = size(stim, 2);
PENALTY = 1e12;

% ---------------------------------------------------------------- stage 1: filter only
[pFilt, losses, ef1] = cascadeStage1Filter(stim, resp, dt, opt.nRandomInits, opt.nRestarts);

% ---------------------------------------------------------------- stage 2: nonlinearity
filt = cascadeFilterSafe(pFilt, nPts, dt);
x    = cascadeConv(stim, filt);
pNL  = cascadeStage2NL(x, resp, opt.nRestarts);

% ---------------------------------------------------------------- stage 3: joint
loss3 = @(v) stage3Loss(v, stim, resp, dt, nPts, PENALTY);
[vJoint, ~, ef3] = cascadeNmRestarts(loss3, [pFilt(:)' pNL(:)'], opt.nRestarts);

params = cascadeCanonical(struct('numFilt',vJoint(1),'tauR',vJoint(2),'tauD',vJoint(3), ...
    'tauP',vJoint(4),'phi',vJoint(5),'alpha',vJoint(6),'beta',vJoint(7), ...
    'gamma',vJoint(8),'epsilon',vJoint(9)), dt);
pred = cascadePredictLN(params, stim, dt);
r2   = computeVarianceExplained(pred, resp);

out = struct('params', params, 'r2PerEpoch', r2(:)', 'r2Mean', mean(r2), 'dt', dt);
if opt.diagnose
    ef = ef3; if isempty(ef), ef = ef1; end
    out.diagnostics = cascadeDiagnose(params, stim, resp, dt, losses, ef, r2, loss3, 'ln');
    if opt.verbose && ~out.diagnostics.ok
        for k = 1:numel(out.diagnostics.warnings)
            fprintf('[cascadeFit] WARNING: %s\n', out.diagnostics.warnings{k});
        end
    end
end
end

% ===================================================================== helpers
function e = stage3Loss(v, stim, resp, dt, nPts, PENALTY)
f = cascadeFilterSafe(v(1:5), nPts, dt);
if isempty(f), e = PENALTY; return; end
g = cascadeConv(stim, f);
pred = v(6) * cascadeNormcdf(v(7)*g + v(8)) + v(9);
e = sum((resp(:) - pred(:)).^2);
if ~isfinite(e), e = PENALTY; end
end
