function [stim, resp, info] = cascadeLoadEpochs(dataPath, varargin)
%CASCADELOADEPOCHS  Load, decimate and preprocess a recording for cascade fitting.
%
%   [stim, resp, info] = cascadeLoadEpochs('cell.mat')
%   [stim, resp, info] = cascadeLoadEpochs('cell.mat', 'dt', 0.01)
%   [stim, resp, info] = cascadeLoadEpochs('cell.mat', 'contract', 'off')
%
% The recording's own declaration decides what happens here -- which variables hold stimulus
% and response, whether the arrays need transposing, the raw sampling interval, what the
% response units are and therefore both its scale factor and whether to rectify. Those rules
% live in scripts/recording.contract.json and are applied by cascadeRecordingContract. This
% function has no opinions about units; it had them once, and so did the Python loader, and
% two copies of the same rules is exactly the drift the contract exists to remove.
%
% Refuses outright on the mistakes that cannot be detected later:
%   - arrays whose shape contradicts the declared orientation
%   - a dt that is not an integer multiple of the raw sampling interval
%   - stim and resp of different sizes
%
% 'contract','off' skips the declaration and falls back to the historical defaults. It exists
% for a recording you are certain about at six in the evening; it warns loudly, and
% cascadePreflight is what should stand between an undeclared recording and a ten-minute job.

p = inputParser;
p.addParameter('dt', 0.01, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('rawDt', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
p.addParameter('metaPath', '', @ischar);
p.addParameter('stimVar', '', @ischar);
p.addParameter('respVar', '', @ischar);
p.addParameter('contract', 'on', @(s) any(strcmpi(s, {'on', 'off'})));
p.addParameter('verbose', true, @islogical);
p.parse(varargin{:});
opt = p.Results;

assert(exist(dataPath, 'file') == 2, 'cascadeLoadEpochs:noFile', 'no such file: %s', dataPath);

metaPath = opt.metaPath;
if isempty(metaPath)
    metaPath = fullfile(fileparts(dataPath), 'meta.json');
end

info = struct('source', dataPath, 'meta', metaPath, 'unresolved', {{}});
useContract = strcmpi(opt.contract, 'on');

% ---- the declaration ------------------------------------------------------------
if useContract
    [plan, report] = cascadeRecordingContract(metaPath);      % strict: rejects throw
    if ~isempty(report.questions)
        % Not a rejection -- a question. The agent is expected to have cleared these with
        % AskUserQuestion and recorded the answers before getting this far.
        qs = strjoin(arrayfun(@(q) q.question, report.questions, 'UniformOutput', false), ' | ');
        error('cascadeLoadEpochs:unresolved', ...
            ['%s leaves %d question(s) unanswered: %s\nAnswer them and record them with ' ...
             'cascadeWriteContract, or pass ''contract'',''off'' to proceed on the ' ...
             'historical defaults.'], metaPath, numel(report.questions), qs);
    end
    info.contract = report.summary;
else
    plan = struct('stimVar', 'stim', 'respVar', 'resp', 'orientation', 'epochs_x_time', ...
                  'transpose', false, 'rawDt', [], 'responseUnits', '', ...
                  'responseScale', 1.0, 'responseKind', '', 'rectify', false);
    info.contract = 'DISABLED';
    info.unresolved{end+1} = ['contract is OFF: variable names, layout, units and ' ...
        'rectification are all assumed rather than declared, and none of them has been ' ...
        'checked against this recording'];
    if opt.verbose
        warning('cascadeLoadEpochs:contractOff', ...
            'contract OFF for %s -- nothing about this recording has been checked', dataPath);
    end
end

% Explicit arguments still win over the declaration; they are how you test a hunch.
stimVar = plan.stimVar; if ~isempty(opt.stimVar); stimVar = opt.stimVar; end
respVar = plan.respVar; if ~isempty(opt.respVar); respVar = opt.respVar; end

% ---- load -----------------------------------------------------------------------
S = load(dataPath);
assert(isfield(S, stimVar) && isfield(S, respVar), 'cascadeLoadEpochs:vars', ...
    '%s contains %s; expected ''%s'' and ''%s''', dataPath, ...
    strjoin(fieldnames(S)', ', '), stimVar, respVar);
stim = double(S.(stimVar));
resp = double(S.(respVar));

if plan.transpose
    stim = stim.';
    resp = resp.';
end

% ---- sampling interval ----------------------------------------------------------
rawDt = opt.rawDt;
info.rawDtFromMetadata = false;
if isempty(rawDt)
    rawDt = plan.rawDt;
    assert(~isempty(rawDt), 'cascadeLoadEpochs:noRawDt', ...
        ['raw sampling interval unknown for %s. Declare ''sample_interval_s'' in meta.json, ' ...
         'or pass ''rawDt'' explicitly rather than guessing.'], dataPath);
    info.rawDtFromMetadata = true;
end
rawDt = double(rawDt);

% ---- shape ----------------------------------------------------------------------
assert(ismatrix(stim) && ismatrix(resp), 'cascadeLoadEpochs:ndims', ...
    'expected (epochs x time) matrices');
assert(isequal(size(stim), size(resp)), 'cascadeLoadEpochs:sizes', ...
    'stim %s and resp %s differ', mat2str(size(stim)), mat2str(size(resp)));
assert(size(stim,1) <= size(stim,2), 'cascadeLoadEpochs:transposed', ...
    ['array is %s after applying the declared layout (%s): more epochs than time points. ' ...
     'If the file really is stored the other way, declare ''orientation'' as ''%s''.'], ...
    mat2str(size(stim)), plan.orientation, ...
    subsref({'time_x_epochs', 'epochs_x_time'}, substruct('{}', {1 + plan.transpose})));

factorF = opt.dt / rawDt;
factor  = round(factorF);
assert(abs(factorF - factor) < 1e-6 && factor >= 1, 'cascadeLoadEpochs:decimation', ...
    ['dt=%g is not an integer multiple of rawDt=%g (factor %.4f); ' ...
     'pick a dt the sampling supports.'], opt.dt, rawDt, factorF);

[nEp, nRaw] = size(stim);
nBins = floor(nRaw / factor);
assert(nBins >= 50, 'cascadeLoadEpochs:tooShort', ...
    'decimating to dt=%g leaves only %d bins per epoch', opt.dt, nBins);
dropped = nRaw - nBins * factor;
if factor > 1
    stim = squeeze(mean(reshape(stim(:, 1:nBins*factor), nEp, factor, nBins), 2));
    resp = squeeze(mean(reshape(resp(:, 1:nBins*factor), nEp, factor, nBins), 2));
    if nEp == 1   % squeeze collapses a singleton first dimension
        stim = stim(:)'; resp = resp(:)';
    end
end

% ---- units ----------------------------------------------------------------------
% Scale AND rectification come from the contract. Both follow the DECLARED unit, never the
% values: rectifying an analog trace destroys the decrement half of an OFF cell's response,
% and failing to rectify a rate spends the fit's dynamic range on negative firing rates.
if plan.responseScale ~= 1.0
    resp = resp * plan.responseScale;
end
rectify = plan.rectify;

info.rawDt = rawDt; info.dt = opt.dt; info.decimationFactor = factor;
info.nEpochs = nEp; info.nBins = nBins; info.droppedRawSamples = dropped;
info.stimVar = stimVar; info.respVar = respVar; info.orientation = plan.orientation;
info.responseUnits = plan.responseUnits; info.responseScale = plan.responseScale;
info.responseKind = plan.responseKind; info.rectify = rectify;
info.stimulusMeanSubtracted = true; info.responseUntouched = (plan.responseScale == 1.0);

stim = stim - mean(stim, 2);      % mean-subtract the stimulus only

if opt.verbose
    src = 'caller'; if info.rawDtFromMetadata, src = 'contract'; end
    extra = ''; if dropped > 0, extra = sprintf(', dropped %d trailing samples', dropped); end
    fprintf('[cascadeFit] %d epochs x %d bins @ dt=%g ms (raw %g us from %s, decimation %dx%s)\n', ...
        nEp, nBins, opt.dt*1e3, rawDt*1e6, src, factor, extra);
    scaleTxt = '';
    if plan.responseScale ~= 1.0
        scaleTxt = sprintf(' x%g->canonical', plan.responseScale);
    end
    fprintf('[cascadeFit] response_units=''%s''%s -> rectify=%d; stimulus mean-subtracted per epoch\n', ...
        plan.responseUnits, scaleTxt, rectify);
    for k = 1:numel(info.unresolved)
        fprintf('[cascadeFit] UNRESOLVED: %s\n', info.unresolved{k});
    end
end
end
