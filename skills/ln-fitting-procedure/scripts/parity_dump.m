function parity_dump(cascadegraph_root, out_mat)
% PARITY_DUMP  Write CascadeGraph reference values for the Python cross-check.
%
%   parity_dump('/path/to/cascadegraph', '/tmp/cg_reference.mat')
%   parity_dump()   % if CascadeGraph is already on your path
%
% Produces the MATLAB side of the Python<->MATLAB parity check: filters, nonlinearity
% outputs, circular convolutions, full LN predictions and row-wise R^2 for a fixed set of
% parameter vectors. Then run scripts/parity_check.py on the same .mat file, which rebuilds
% all of it with cascade_fit.py and reports the maximum absolute difference.
%
% The stimulus is written out rather than regenerated on the Python side, so both languages
% see bit-identical inputs and any difference is genuinely the model code.

if nargin < 2 || isempty(out_mat)
    out_mat = fullfile(tempdir, 'cg_reference.mat');
end
if nargin >= 1 && ~isempty(cascadegraph_root)
    addpath(genpath(cascadegraph_root));
end
assert(exist('ParamFilterNode','class') == 8, 'parity_dump:noCascadeGraph', ...
    ['CascadeGraph is not on the MATLAB path. Either call this as ' ...
     'parity_dump(''/path/to/cascadegraph'') or run addpath(genpath(...)) yourself first.']);
addpath(fileparts(mfilename('fullpath')));                       % this scripts/ folder
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab'));   % the fitting layer

dt = 0.01;
nPts = 400;
nEpochs = 3;

% Deterministic stimulus, no RNG: identical every run and on both sides.
t = (1:nPts);
stim = zeros(nEpochs, nPts);
for e = 1:nEpochs
    stim(e,:) = sin(2*pi*t/37.0 + e) + 0.5*cos(2*pi*t/11.0 - 2*e) + 0.25*sin(2*pi*t/5.0 + 3*e);
end
stim = stim - mean(stim, 2);          % mean-subtract per epoch, as the pipeline does

% Parameter sets: typical, plus the awkward corners that expose convention drift.
P = [ 4.0   0.025  0.045  0.065   35.0
      1.0   0.005  0.200  0.010 -180.0
      9.5   0.090  0.012  0.095  179.0
    250.0   0.030  0.050  0.070    0.0      % huge numFilt: rise becomes a step
      5.0   0.018  0.035  0.050  120.0 ];

nl_params = [ -55.0  1.0577364234585758  -0.4  18.0
               45.0  1.9606461901796066   0.9 -62.0 ];
nl_x = linspace(-4, 4, 17);

filters = cell(size(P,1),1);
convs   = cell(size(P,1),1);
preds   = cell(size(P,1),1);
r2s     = cell(size(P,1),1);
nls     = cell(size(nl_params,1),1);

for i = 1:size(P,1)
    p.numFilt = P(i,1); p.tauR = P(i,2); p.tauD = P(i,3); p.tauP = P(i,4); p.phi = P(i,5);
    filters{i} = ParamFilterNode.getFilterWithParams(p, nPts, dt);
    convs{i}   = ParamFilterNode.processTempParams(p, stim, dt);
    q = nl_params(1,:);
    preds{i}   = SigmoidNlNode.processTempParams(q(:), convs{i});
    r2s{i}     = computeVarianceExplained(preds{i}, stim);   % arbitrary "measured", exercises the metric
end

for j = 1:size(nl_params,1)
    nls{j} = SigmoidNlNode.processTempParams(nl_params(j,:)', nl_x);
end

% --- GLM free-running and two-arm, so the cross-check covers every model in the family ---
glmP = [3.0 0.030 0.050 0.080 -20.0];
gf   = ParamFilterNode.getFilterWithParams( ...
         struct('numFilt',glmP(1),'tauR',glmP(2),'tauD',glmP(3),'tauP',glmP(4),'phi',glmP(5)), nPts, dt);
gx   = real(ifft(fft(stim') .* fft(gf)))';
glmNL = [-70.0 1.6 -0.3 25.0];
glmFb = [-0.0137 0.06]; nFb = 30;
ref.glmP = glmP; ref.glmNL = glmNL; ref.glmFb = glmFb; ref.nFb = nFb;
ref.glmPred = cascadeFreeRun(gx, glmNL(1), glmNL(2), glmNL(3), glmNL(4), glmFb(1), glmFb(2), nFb, dt);

taP = struct('numFilt1',4.0,'tauR1',0.028,'tauD1',0.050,'tauP1',0.070,'phi1',25.0, ...
             'numFilt2',3.0,'tauR2',0.014,'tauD2',0.028,'tauP2',0.045,'phi2',-95.0, ...
             'alpha1',65.0,'beta1',1.7,'gamma1',-0.35,'epsilon1',-30.0, ...
             'alpha2',3.0,'beta2',3.0,'gamma2',0.5,'epsilon2',0.0);
ref.twoArmFields = fieldnames(taP);
ref.twoArmVals   = cell2mat(struct2cell(taP));
ref.twoArmPred   = cascadePredictTwoArm(taP, stim, dt);

ref.dt = dt; ref.nPts = nPts; ref.stim = stim;
ref.P = P; ref.nl_params = nl_params; ref.nl_x = nl_x;
ref.filters = filters; ref.convs = convs; ref.preds = preds; ref.r2s = r2s; ref.nls = nls;
ref.matlab_version = version;

save(out_mat, '-struct', 'ref', '-v7');
fprintf('wrote %s\n', out_mat);
fprintf('  %d parameter sets, %d epochs x %d points, dt=%g\n', size(P,1), nEpochs, nPts, dt);
end
