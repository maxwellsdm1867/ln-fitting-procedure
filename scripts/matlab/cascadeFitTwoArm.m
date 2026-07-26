function out = cascadeFitTwoArm(stim, resp, dt, varargin)
%CASCADEFITTWOARM  Staged fit of CascadeGraph's TwoArmLnHyperNode topology (18 params).
%
%   out = cascadeFitTwoArm(stim, resp, dt)
%
%   prediction = NL1( filter1(stim) + NL2( filter2(stim) ) )
%
% ONE LINEAR ARM and one nonlinear arm summed, then a nonlinearity -- not two symmetric LN
% arms. Arm 1 has no nonlinearity and no gain of its own.
%
% epsilon2 is HELD AT 0 because (epsilon2, gamma1) is an exact degeneracy: epsilon2 shifts
% arm 2's output, which shifts the sum, which gamma1 shifts back. Leaving both free adds a
% flat ridge the simplex wanders along instead of shaping the second arm -- measured at 0.071
% R^2 on synthetic two-arm data, so this is not a tidiness point.
%
% alpha2 IS identifiable, and only because ParamFilterNode normalises to unit peak: arm 1 has
% no free scale, so alpha2 weighs arm 2 against a fixed reference. Drop that normalisation and
% the arms trade amplitude and neither means anything.
%
% Identifiability also needs the arms COMPARABLE in magnitude. If either dominates,
% NL1(NL2(f2*s)) collapses to a single filter with a composed static nonlinearity and a
% one-arm LN fits it just as well.

p = inputParser;
p.addParameter('nRandomInits', 20, @isscalar);
p.addParameter('nRestarts', 10, @isscalar);
p.addParameter('nArm2Starts', 6, @isscalar);
p.addParameter('seed', 0, @isscalar);
p.addParameter('verbose', true, @islogical);
p.parse(varargin{:});
opt = p.Results;
assert(exist('ParamFilterNode','class') == 8, 'cascadeFitTwoArm:noCascadeGraph', ...
    'CascadeGraph is not on the path.');

rng(opt.seed);
PENALTY = 1e12; nPts = size(stim,2);

% ---- seed arm 1 and NL1 from the single-arm LN fit. Starting arm 2 alongside an unconverged
%      arm 1 gives the optimizer two ways to explain the same variance.
ln = cascadeFitLN(stim, resp, dt, 'nRandomInits', opt.nRandomInits, ...
                  'nRestarts', opt.nRestarts, 'seed', opt.seed, 'diagnose', false, ...
                  'verbose', false);
lp = ln.params;
base = [lp.numFilt lp.tauR lp.tauD lp.tauP lp.phi];
nl1  = [lp.alpha lp.beta lp.gamma lp.epsilon];
x1sd = std(reshape(cascadeConv(stim, cascadeFilterSafe(base, nPts, dt)), [], 1));

RANGES = [1 10; 0.005 0.1; 0.005 0.2; 0.01 0.1; -180 180];
loss = @(v) twoArmLoss(v, stim, resp, dt, nPts, PENALTY);

bestV = []; bestF = inf;
for i = 1:opt.nArm2Starts
    f2 = RANGES(:,1)' + rand(1,5) .* diff(RANGES,1,2)';
    a2 = (0.2 + 1.8*rand) * x1sd * sign(randn);      % comparable to arm 1, either sign
    v0 = [base f2 nl1 a2, 2.0/max(x1sd, eps), 2*rand-1];
    [v, fv] = cascadeNmRestarts(loss, v0, opt.nRestarts);
    if fv < bestF, bestV = v; bestF = fv; end
end

params = unpackTwoArm(bestV);
pred = cascadePredictTwoArm(params, stim, dt);
r2 = computeVarianceExplained(pred, resp);

out = struct('params', params, 'r2PerEpoch', r2(:)', 'r2Mean', mean(r2), 'dt', dt, ...
             'lnR2Mean', ln.r2Mean, 'gainOverLN', mean(r2) - ln.r2Mean);
out.diagnostics = cascadeDiagnose(params, stim, resp, dt, [], [], r2, loss, 'twoarm');
if opt.verbose && ~out.diagnostics.ok
    for k = 1:numel(out.diagnostics.warnings)
        fprintf('[cascadeFit] WARNING: %s\n', out.diagnostics.warnings{k});
    end
end
lnPred    = cascadePredictLN(lp, stim, dt);
out.bic   = cascadeBic(resp, pred,   17);
out.lnBic = cascadeBic(resp, lnPred,  9);

if opt.verbose
    fprintf('[cascadeFit] two-arm R^2 %.4f vs single-arm LN %.4f (gain %+.4f)\n', ...
            out.r2Mean, ln.r2Mean, out.gainOverLN);
    fprintf('[cascadeFit] BIC two-arm %.0f vs LN %.0f (lower wins; R^2 alone always favours the bigger model)\n', ...
            out.bic, out.lnBic);
end
end

function p = unpackTwoArm(v)
p = struct('numFilt1',v(1),'tauR1',abs(v(2)),'tauD1',abs(v(3)),'tauP1',v(4),'phi1',v(5), ...
           'numFilt2',v(6),'tauR2',abs(v(7)),'tauD2',abs(v(8)),'tauP2',v(9),'phi2',v(10), ...
           'alpha1',v(11),'beta1',v(12),'gamma1',v(13),'epsilon1',v(14), ...
           'alpha2',v(15),'beta2',v(16),'gamma2',v(17),'epsilon2',0);
end

function e = twoArmLoss(v, stim, resp, dt, nPts, PENALTY)
p = unpackTwoArm(v);
pred = cascadePredictTwoArm(p, stim, dt);
if isempty(pred), e = PENALTY; return; end
e = sum((resp(:) - pred(:)).^2);
if ~isfinite(e), e = PENALTY; end
end
