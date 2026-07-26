function [stim, resp, info] = cascadeLoadEpochs(dataPath, varargin)
%CASCADELOADEPOCHS  Load, decimate and preprocess a recording for cascade fitting.
%
%   [stim, resp, info] = cascadeLoadEpochs('cell.mat')
%   [stim, resp, info] = cascadeLoadEpochs('cell.mat', 'dt', 0.01, 'rawDt', 1e-4)
%
% Infers the setup rather than making you restate it: the raw sampling interval is read from
% a sibling meta.json ('sample_interval_s') when not supplied, and response_units decides
% whether rectification would be appropriate. The setup actually used is printed once and
% returned in INFO, because a decimation or orientation mistake is invisible downstream and
% expensive.
%
% Refuses outright on the mistakes that cannot be detected later:
%   - arrays that are (time x epochs) rather than (epochs x time)
%   - a dt that is not an integer multiple of the raw sampling interval
%   - stim and resp of different sizes
%
% Anything it cannot resolve -- units missing or unrecognised, no metadata at all -- is
% returned in info.unresolved and printed. Known and checked is quiet; unknown is loud.
%
% Expects DATAPATH to be a .mat containing variables `stim` and `resp`, each (epochs x time).

p = inputParser;
p.addParameter('dt', 0.01, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('rawDt', [], @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x > 0));
p.addParameter('metaPath', '', @ischar);
p.addParameter('stimVar', 'stim', @ischar);
p.addParameter('respVar', 'resp', @ischar);
p.addParameter('verbose', true, @islogical);
p.parse(varargin{:});
opt = p.Results;

assert(exist(dataPath, 'file') == 2, 'cascadeLoadEpochs:noFile', 'no such file: %s', dataPath);
S = load(dataPath);
assert(isfield(S, opt.stimVar) && isfield(S, opt.respVar), 'cascadeLoadEpochs:vars', ...
    '%s contains %s; expected variables ''%s'' and ''%s''', dataPath, ...
    strjoin(fieldnames(S)', ', '), opt.stimVar, opt.respVar);
stim = double(S.(opt.stimVar));
resp = double(S.(opt.respVar));

info = struct('source', dataPath, 'unresolved', {{}});
metaPath = opt.metaPath;
if isempty(metaPath)
    metaPath = fullfile(fileparts(dataPath), 'meta.json');
end
meta = struct();
if exist(metaPath, 'file') == 2
    try
        meta = jsondecode(fileread(metaPath));
    catch
        meta = struct();
    end
end
info.meta = metaPath;

rawDt = opt.rawDt;
info.rawDtFromMetadata = false;
if isempty(rawDt)
    assert(isfield(meta, 'sample_interval_s'), 'cascadeLoadEpochs:noRawDt', ...
        ['raw sampling interval unknown: no ''sample_interval_s'' in %s. ' ...
         'Pass ''rawDt'' explicitly rather than guessing.'], metaPath);
    rawDt = meta.sample_interval_s;
    info.rawDtFromMetadata = true;
end
rawDt = double(rawDt);

assert(ismatrix(stim) && ismatrix(resp), 'cascadeLoadEpochs:ndims', ...
    'expected (epochs x time) matrices');
assert(isequal(size(stim), size(resp)), 'cascadeLoadEpochs:sizes', ...
    'stim %s and resp %s differ', mat2str(size(stim)), mat2str(size(resp)));
assert(size(stim,1) <= size(stim,2), 'cascadeLoadEpochs:transposed', ...
    ['array is %s: more epochs than time points, which almost always means it is ' ...
     'transposed. Expected (epochs x time).'], mat2str(size(stim)));

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

units = '';
if isfield(meta, 'response_units'), units = meta.response_units; end
knownRate   = {'spikes/s','spikes/sec','Hz','sp/s'};
knownAnalog = {'mV','pA','nA','uA','V','A'};
if any(strcmp(units, knownRate))
    rectify = true;
elseif any(strcmp(units, knownAnalog))
    rectify = false;
else
    rectify = false;
    info.unresolved{end+1} = sprintf(['response_units is ''%s'', which is not a recognised ' ...
        'rate (%s) or analog (%s) unit. Assuming ANALOG, so rectify=false. If this is a ' ...
        'firing rate, say so -- rectifying an analog trace, or failing to rectify a rate, ' ...
        'changes which model you are fitting.'], units, strjoin(knownRate,', '), ...
        strjoin(knownAnalog,', '));
end
if isempty(fieldnames(meta))
    info.unresolved{end+1} = sprintf(['no metadata at %s: cell type, protocol and units are ' ...
        'unknown, so nothing here has been checked against the recording''s own description.'], ...
        metaPath);
end

info.rawDt = rawDt; info.dt = opt.dt; info.decimationFactor = factor;
info.nEpochs = nEp; info.nBins = nBins; info.droppedRawSamples = dropped;
info.responseUnits = units; info.rectify = rectify;
info.stimulusMeanSubtracted = true; info.responseUntouched = true;

stim = stim - mean(stim, 2);      % mean-subtract the stimulus only

if opt.verbose
    src = 'caller'; if info.rawDtFromMetadata, src = 'metadata'; end
    extra = ''; if dropped > 0, extra = sprintf(', dropped %d trailing samples', dropped); end
    fprintf('[cascadeFit] %d epochs x %d bins @ dt=%g ms (raw %g us from %s, decimation %dx%s)\n', ...
        nEp, nBins, opt.dt*1e3, rawDt*1e6, src, factor, extra);
    fprintf('[cascadeFit] response_units=''%s'' -> rectify=%d; stimulus mean-subtracted per epoch, response in native units\n', ...
        units, rectify);
    for k = 1:numel(info.unresolved)
        fprintf('[cascadeFit] UNRESOLVED: %s\n', info.unresolved{k});
    end
end
end
