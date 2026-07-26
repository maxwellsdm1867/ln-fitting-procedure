function d = cascadeDiagnose(params, stim, resp, dt, losses, exitflag, r2, lossFn, model)
%CASCADEDIAGNOSE  Run every mechanical check and return a verdict.
%
% The point is not more numbers to read. It is that the conventions, the degeneracies and the
% optimizer's own status are checked automatically, so attention goes to the fit and the data
% rather than to bookkeeping. d.ok true with empty d.warnings means the mechanical failure
% modes are ruled out and what is left is science.
if nargin < 9, model = 'ln'; end
w = {}; c = struct();

c.converged = exitflag;
if ~isempty(exitflag) && exitflag == 0
    w{end+1} = 'optimizer did not converge: fminsearch exhausted MaxFunEvals (exitflag 0) rather than settling';
end

if ~isempty(losses)
    best = min(losses); nClose = sum(losses <= best*1.01);
    c.startAgreement = [nClose numel(losses)];
    if nClose < 2
        w{end+1} = sprintf(['only 1 of %d starts reached the best loss: the answer depends on ' ...
            'the seed, so report the dispersion or add starts'], numel(losses));
    end
end

if strcmp(model,'twoarm')
    v = [params.numFilt1 params.tauR1 params.tauD1 params.tauP1 params.phi1 ...
         params.numFilt2 params.tauR2 params.tauD2 params.tauP2 params.phi2 ...
         params.alpha1 params.beta1 params.gamma1 params.epsilon1 ...
         params.alpha2 params.beta2 params.gamma2];      % epsilon2 is held at 0
    filtParams = struct('numFilt',params.numFilt1,'tauR',params.tauR1,'tauD',params.tauD1, ...
                        'tauP',params.tauP1,'phi',params.phi1);
else
    v = [params.numFilt params.tauR params.tauD params.tauP params.phi ...
         params.alpha params.beta params.gamma params.epsilon];
    if strcmp(model,'glm'), v = [v params.a_fb params.tau_fb]; end
    filtParams = params;
end
f0 = lossFn(v); worst = inf;
for i = 1:numel(v)
    step = max(abs(v(i))*0.02, 1e-4);
    up = v; up(i) = up(i)+step; dn = v; dn(i) = dn(i)-step;
    worst = min(worst, min(lossFn(up), lossFn(dn)) - f0);
end
c.localOptimumWorstRel = worst / max(abs(f0), eps);
if worst < 0
    w{end+1} = sprintf(['not at a local minimum: perturbing a parameter lowers the loss by ' ...
        '%.1e relative, so the search stopped early'], abs(c.localOptimumWorstRel));
end

switch model
    case 'glm',    pred = cascadePredictGLM(params, stim, dt);
    case 'twoarm', pred = cascadePredictTwoArm(params, stim, dt);
    otherwise,     pred = cascadePredictLN(params, stim, dt);
end
gap  = abs(mean(computeVarianceExplained(pred, resp)) - mean(r2));
c.roundtripGap = gap;
if gap > 0.01
    w{end+1} = 'reported parameters do not reproduce the fit -- do not report them yet';
end

nPts = size(stim,2);
filt = ParamFilterNode.getFilterWithParams(filtParams, nPts, dt);
imp = zeros(1, nPts); at = floor(nPts/4); imp(at) = 1;
y = real(ifft(fft(imp') .* fft(filt)))';
[~, k] = max(abs(y)); lag = k - at;
if lag > nPts/2, lag = lag - nPts; end
c.causalityLagBins = lag;
if lag < 0
    w{end+1} = sprintf(['filter is acausal (impulse response peaks at lag %d): the model is ' ...
        'predicting from future stimulus'], lag);
end

nP = min(60, nPts);
S = fft(stim, [], 2); R = fft(resp - mean(resp,2), [], 2);
F = mean(R .* conj(S), 1); F(1) = 0;
fnp = real(ifft(F)); fnp = fnp(1:nP);
cc = abs(cascadeCorr(filt(1:nP), fnp(:)));
c.filterVsXcorr = cc;
% For a GLM the cross-correlation estimates the CLOSED-LOOP effective filter, not the model's
% front end, so the two legitimately differ and a low value is not evidence of a bad fit.
if cc < 0.8 && ~any(strcmp(model,{'glm','twoarm'}))
    w{end+1} = sprintf(['fitted filter correlates only %.2f with the cross-correlation ' ...
        'estimate: likely a local minimum'], cc);
end

% numFilt is only identifiable while the rise is resolvable at this sampling. The 10-90%
% rise of 1/(1+(tauR/t)^n) is ~ tauR*2*ln(9)/n, so above roughly 22*tauR/dt the discrete
% filter stops changing at all (bit-identical for n = 500 vs 2187 at dt = 10 ms). A large
% fitted numFilt is a flat direction, not a sharp rise, and is a bound not an estimate.
if isfield(params,'numFilt1'), nf = params.numFilt1; tr = params.tauR1;
else, nf = params.numFilt; tr = params.tauR; end
nCrit = 22 * abs(tr) / dt;
c.numFiltIdentifiableBelow = nCrit;
if nf > nCrit
    w{end+1} = sprintf(['numFilt = %.0f is above the resolvable limit for this sampling ' ...
        '(~%.0f = 22*tauR/dt): the rise is faster than one bin, so the filter is unchanged ' ...
        'above it and the value is a flat direction, not an estimate. Report it as >= %.0f, ' ...
        'or refit at finer dt to resolve it'], nf, nCrit, nCrit);
end

c.perEpochR2Spread = max(r2) - min(r2);
if c.perEpochR2Spread > 0.15
    w{end+1} = sprintf(['per-epoch R^2 spread is %.2f: suspect one bad trial rather than the ' ...
        'model'], c.perEpochR2Spread);
end

d = struct('ok', isempty(w), 'warnings', {w}, 'checks', c);
end
