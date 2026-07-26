function proposal = cascadeInferContract(dataPath)
%CASCADEINFERCONTRACT  Read a recording and PROPOSE its contract. Never assumes.
%
%   proposal = cascadeInferContract('cell.mat')
%
% The contract has to exist before anything can be fitted, but making the scientist write it
% from scratch is how you get a directory of recordings with no meta.json. So: infer what the
% file can actually tell us, propose the rest, and mark honestly which is which. The caller
% (the agent) confirms with AskUserQuestion and then calls cascadeWriteContract.
%
% PROPOSAL.fields     struct -- the proposed meta.json
% PROPOSAL.confidence struct -- per field: 'certain' | 'likely' | 'must_ask'
% PROPOSAL.evidence   struct -- per field: what the file actually showed
% PROPOSAL.mustAsk    cellstr -- fields nothing in the file can settle
%
% The confidence levels are the point. 'certain' is read off the file. 'likely' is a heuristic
% that is usually right and is still shown for confirmation. 'must_ask' means the file is
% genuinely silent and the answer changes the numbers -- notably sample_interval_s, which no
% array can reveal, and response units, where the magnitude only hints.

    assert(exist(dataPath, 'file') == 2, 'cascadeInfer:noFile', 'no such file: %s', dataPath);

    vars = readVars(dataPath);
    allNames = fieldnames(vars);
    % Scalars cannot be stim or resp, but they are exactly where a sampling interval hides,
    % so the two searches get different candidate lists rather than one filtered list.
    names = allNames(cellfun(@(n) ~isscalar(vars.(n)), allNames));
    assert(~isempty(names), 'cascadeInfer:empty', '%s holds no numeric arrays', dataPath);

    fields = struct(); conf = struct(); ev = struct();

    % ---- which variable is which ------------------------------------------
    [stimVar, stimHow] = pick(names, {'stim', 'stimulus', 'lightstim', 'light', 'input'});
    [respVar, respHow] = pick(setdiff(names, {stimVar}), ...
                              {'resp', 'response', 'voltage', 'current', 'v', 'i', 'output'});
    fields.stim_var = stimVar; conf.stim_var = stimHow;
    ev.stim_var = sprintf('file holds: %s', strjoin(names', ', '));
    fields.resp_var = respVar; conf.resp_var = respHow;
    ev.resp_var = ev.stim_var;

    % ---- layout -----------------------------------------------------------
    % An epoch is far longer than an experiment has epochs, so the long axis is time. This is
    % reliable enough to propose and never safe enough to apply silently -- a 3x3 array, or a
    % single very short epoch, breaks it.
    sz = size(vars.(respVar));
    if numel(sz) == 2 && sz(1) > sz(2)
        fields.orientation = 'time_x_epochs';
    else
        fields.orientation = 'epochs_x_time';
    end
    ratio = max(sz) / max(1, min(sz));
    if ratio >= 10
        conf.orientation = 'likely';
    else
        conf.orientation = 'must_ask';
    end
    ev.orientation = sprintf('%s is %dx%d, so the long axis is %s (ratio %.0f:1)', ...
        respVar, sz(1), sz(2), ternary(sz(1) > sz(2), 'dim 1', 'dim 2'), ratio);

    % ---- sampling interval ------------------------------------------------
    % Nothing in an array of numbers reveals its sampling rate. If the file happens to carry
    % it, take it; otherwise this is the field that must be asked, every time.
    [dtVal, dtHow, dtWhy] = findSampleInterval(vars, allNames);
    fields.sample_interval_s = dtVal;
    conf.sample_interval_s = dtHow;
    ev.sample_interval_s = dtWhy;

    % ---- response units ---------------------------------------------------
    % Magnitude hints and nothing more. A patch trace in mV and one in pA differ by a factor
    % that overlaps; a rate is non-negative but so is a rectified current. Propose, never set.
    r = double(vars.(respVar)(:));
    r = r(isfinite(r));
    [unitGuess, unitWhy] = guessUnits(r);
    fields.response_units = unitGuess;
    conf.response_units = 'must_ask';
    ev.response_units = unitWhy;

    fields.stimulus_units = '';
    conf.stimulus_units = 'must_ask';
    ev.stimulus_units = ['arbitrary by design -- contrast, pixel intensity, R*/cone/s and raw ' ...
                         'DAC counts are all legitimate and none can be read off the values'];

    mustAsk = {};
    cf = fieldnames(conf);
    for i = 1:numel(cf)
        if strcmp(conf.(cf{i}), 'must_ask'); mustAsk{end+1} = cf{i}; end %#ok<AGROW>
    end

    proposal = struct('source', dataPath, 'fields', fields, 'confidence', conf, ...
                      'evidence', ev, 'mustAsk', {mustAsk});
end

% ---------------------------------------------------------------------------
function v = readVars(p)
    v = struct();
    [~, ~, ext] = fileparts(lower(p));
    if strcmp(ext, '.mat')
        try
            v = load(p);
        catch
            fh = h5info(p);                     % v7.3
            for i = 1:numel(fh.Datasets)
                n = fh.Datasets(i).Name;
                v.(matlab.lang.makeValidName(n)) = h5read(p, ['/' n])';
            end
        end
    else
        error('cascadeInfer:format', ...
            '%s is not a .mat -- the MATLAB path reads .mat recordings', p);
    end
    f = fieldnames(v);
    for i = 1:numel(f)
        if ~isnumeric(v.(f{i}))
            v = rmfield(v, f{i});      % scalars stay: sample_interval_s / Fs live there
        end
    end
end

function [name, how] = pick(names, preferred)
    name = ''; how = 'must_ask';
    for i = 1:numel(preferred)
        hit = names(strcmpi(names, preferred{i}));
        if ~isempty(hit); name = hit{1}; how = 'certain'; return; end
    end
    for i = 1:numel(preferred)
        hit = names(contains(lower(names), preferred{i}));
        if ~isempty(hit); name = hit{1}; how = 'likely'; return; end
    end
    if ~isempty(names); name = names{1}; end
end

function [val, how, why] = findSampleInterval(vars, names)
    val = []; how = 'must_ask';
    why = ['no sampling interval in the file -- an array of numbers cannot reveal its own ' ...
           'sample rate, and guessing it scales the whole time axis'];
    for i = 1:numel(names)
        n = lower(names{i});
        if contains(n, 'interval') || contains(n, 'dt') || contains(n, 'samplerate') || ...
           contains(n, 'fs')
            x = double(vars.(names{i}));
            if isscalar(x) && isfinite(x) && x > 0
                if x > 1        % looks like a RATE, not an interval
                    val = 1 / x; why = sprintf('%s = %g in the file, read as a RATE -> %g s', ...
                                               names{i}, x, val);
                else
                    val = x; why = sprintf('%s = %g s in the file', names{i}, x);
                end
                how = 'likely';   % still confirm: rate-vs-interval is a coin flip on the name
                return
            end
        end
    end
end

function [unit, why] = guessUnits(r)
    if isempty(r)
        unit = ''; why = 'response is empty'; return
    end
    lo = min(r); hi = max(r); span = hi - lo;
    negFrac = mean(r < 0);
    isInt = all(abs(r - round(r)) < 1e-9);
    if negFrac < 1e-9 && isInt && hi > 1
        unit = 'spikes/s';
        why = sprintf(['all values non-negative integers in [%g, %g] -- consistent with a ' ...
                       'firing rate, but a rectified current looks the same'], lo, hi);
    elseif abs(hi) < 1 && abs(lo) < 1 && span < 1
        unit = 'V';
        why = sprintf(['values in [%g, %g], span %g -- small enough to be volts rather than ' ...
                       'millivolts, which is a 1000x error if wrong'], lo, hi, span);
    elseif span > 50
        unit = 'pA';
        why = sprintf(['values in [%g, %g], span %g -- large span suggests current in pA, ' ...
                       'though a large voltage deflection looks identical'], lo, hi, span);
    else
        unit = 'mV';
        why = sprintf('values in [%g, %g], span %g -- consistent with millivolts', lo, hi, span);
    end
end

function out = ternary(c, a, b)
    if c; out = a; else; out = b; end
end
