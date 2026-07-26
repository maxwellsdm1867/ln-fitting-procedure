function [plan, report] = cascadeRecordingContract(metaPath, varargin)
%CASCADERECORDINGCONTRACT  Decide whether a recording can be fitted, and how.
%
%   [plan, report] = cascadeRecordingContract('meta.json')
%   [plan, report] = cascadeRecordingContract('meta.json', 'strict', false)
%
% The rules are NOT in this file. They are in scripts/recording.contract.json, which is data:
% the canonical layout and units, the required fields, the sane range for the sampling
% interval, and the unit table that decides scaling and rectification. This function only
% applies them, so there is exactly one place to change what the skill accepts.
%
% Three outcomes, and the distinction is the whole point:
%
%   REJECT   -- the recording cannot be fitted as declared. report.rejects lists every reason
%               at once, not the first one, because reporting them one per run turns a
%               five-minute fix into five rounds of it.
%   ASK      -- something is genuinely ambiguous and the answer changes the result.
%               report.questions carries what to ask. DO NOT GUESS: a response left in V fits
%               happily and puts every fitted amplitude 1000x from the thresholds in this
%               skill, and nothing in R^2 shows it. Ask the scientist, with AskUserQuestion.
%   ACCEPT   -- plan says exactly what to apply: which variables, whether to transpose, the
%               raw sampling interval, the response scale factor, and whether to rectify.
%
% PLAN fields: stimVar, respVar, orientation, transpose, rawDt, responseUnits, responseScale,
% responseKind, rectify.

    p = inputParser;
    p.addParameter('strict', true, @islogical);
    p.parse(varargin{:});

    here = fileparts(mfilename('fullpath'));                       % .../scripts/matlab
    contractPath = fullfile(here, '..', 'recording.contract.json');
    assert(exist(contractPath, 'file') == 2, 'cascadeContract:noContract', ...
        'recording contract not found at %s', contractPath);
    C = jsondecode(fileread(contractPath));

    rejects = {};
    questions = struct('field', {}, 'question', {}, 'options', {}, 'why', {});

    % ---- read the declaration -------------------------------------------------
    meta = struct();
    if exist(metaPath, 'file') ~= 2
        rejects{end+1} = sprintf(['no meta.json at %s: nothing about this recording is ' ...
            'declared, so nothing about it has been checked'], metaPath);
    else
        try
            meta = jsondecode(fileread(metaPath));
        catch ME
            rejects{end+1} = sprintf('%s is not readable JSON (%s)', metaPath, ME.message);
        end
    end

    % ---- variable names and layout -------------------------------------------
    plan = struct();
    plan.stimVar    = getOr(meta, 'stim_var', C.fields.stim_var.default);
    plan.respVar    = getOr(meta, 'resp_var', C.fields.resp_var.default);
    plan.orientation = getOr(meta, 'orientation', C.fields.orientation.default);

    allowed = cellstr(C.fields.orientation.enum);
    if ~any(strcmp(plan.orientation, allowed))
        rejects{end+1} = sprintf('orientation is ''%s''; expected one of: %s', ...
            plan.orientation, strjoin(allowed, ', '));
        plan.orientation = C.fields.orientation.default;
    end
    plan.transpose = strcmp(plan.orientation, 'time_x_epochs');

    % ---- sampling interval ----------------------------------------------------
    rawDt = [];
    if isfield(meta, 'sample_interval_s')
        v = meta.sample_interval_s;
        if ~isnumeric(v) || ~isscalar(v) || ~isfinite(v)
            rejects{end+1} = sprintf('sample_interval_s is not a finite number (got %s)', ...
                strtrim(evalc('disp(v)')));
        else
            rawDt = double(v);
            lo = C.fields.sample_interval_s.sane_range(1);
            hi = C.fields.sample_interval_s.sane_range(2);
            if rawDt < lo || rawDt > hi
                hint = '';
                if rawDt > hi
                    hint = sprintf([' -- %g looks like milliseconds or a sample RATE; in ' ...
                        'seconds that would be %g or %g'], rawDt, rawDt/1000, 1/rawDt);
                end
                rejects{end+1} = sprintf(...
                    'sample_interval_s = %g s is outside the sane range [%g, %g]%s', ...
                    rawDt, lo, hi, hint);
            end
        end
    else
        rejects{end+1} = ['no ''sample_interval_s'': the raw sampling interval is unknown, ' ...
            'and guessing it silently changes the decimation factor'];
    end
    plan.rawDt = rawDt;

    % ---- response units: scaling AND rectification ----------------------------
    % Both follow the DECLARED unit, never the values. Rectifying an analog trace destroys the
    % decrement half of an OFF cell's response; failing to rectify a rate spends the fit's
    % dynamic range on negative firing rates. Neither shows up in R^2.
    tbl = C.response_units_table;
    units = getOr(meta, 'response_units', '');
    plan.responseUnits = units;
    plan.responseScale = 1.0;
    plan.rectify = false;
    plan.responseKind = '';

    idx = 0;
    for i = 1:numel(tbl)
        if strcmp(tbl(i).unit, units); idx = i; break; end
    end

    if idx > 0
        plan.responseScale = double(tbl(idx).to_canonical);
        plan.rectify = logical(tbl(idx).rectify);
        plan.responseKind = tbl(idx).kind;
    elseif isempty(units)
        questions(end+1) = mkq('response_units', ...
            'What are the units of the stored response?', ...
            {'mV (analog voltage)', 'pA (analog current)', 'spikes/s (firing rate)'}, ...
            ['response_units is not declared. It decides BOTH the scale factor and whether ' ...
             'to rectify, and getting either wrong is invisible in R^2.']);
    else
        known = cell(1, numel(tbl));
        for i = 1:numel(tbl); known{i} = tbl(i).unit; end
        questions(end+1) = mkq('response_units', ...
            sprintf('Response units are declared as ''%s'', which the contract does not know. What is it?', units), ...
            {'a voltage -- give the factor to mV', 'a current -- give the factor to pA', ...
             'a firing rate in spikes/s'}, ...
            sprintf(['unit ''%s'' is not in the contract''s table (%s). Either declare ' ...
                     '''response_scale'' explicitly, or add the unit to the contract.'], ...
                    units, strjoin(known, ', ')));
    end

    % An explicit response_scale is the escape hatch and overrides the table.
    if isfield(meta, 'response_scale') && ~isempty(meta.response_scale)
        v = meta.response_scale;
        if ~isnumeric(v) || ~isscalar(v) || ~isfinite(v)
            rejects{end+1} = 'response_scale is not a finite number';
        else
            plan.responseScale = double(v);
            % An explicit factor answers the scaling question, but NOT rectification.
            if idx == 0
                keep = true(1, numel(questions));
                for i = 1:numel(questions)
                    if strcmp(questions(i).field, 'response_units'); keep(i) = false; end
                end
                questions = questions(keep);
                questions(end+1) = mkq('rectify', ...
                    'Is the response a firing rate (rectify) or an analog trace (do not)?', ...
                    {'analog -- do not rectify', 'firing rate -- rectify'}, ...
                    ['response_scale was given explicitly, so the scaling is settled, but ' ...
                     'rectification still follows the KIND of signal and cannot be inferred ' ...
                     'from a number.']);
            end
        end
    end

    % ---- verdict --------------------------------------------------------------
    report = struct('rejects', {rejects}, 'questions', questions, ...
                    'contract', contractPath, 'source', metaPath);
    report.accepted = isempty(rejects) && isempty(questions);
    report.summary = describe(plan);

    if p.Results.strict && ~isempty(rejects)
        error('cascadeContract:rejected', '%s does not satisfy the recording contract:\n%s', ...
            metaPath, strjoin(cellfun(@(s) ['  - ' s], rejects, 'UniformOutput', false), newline));
    end
end

% ---------------------------------------------------------------------------
function v = getOr(s, name, dflt)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        v = s.(name);
        if isstring(v); v = char(v); end
    else
        v = dflt;
        if isstring(v); v = char(v); end
    end
end

function q = mkq(field, question, options, why)
    q = struct('field', field, 'question', question, 'options', {options}, 'why', why);
end

function s = describe(plan)
    bits = {sprintf('vars %s/%s', plan.stimVar, plan.respVar), plan.orientation};
    if ~isempty(plan.rawDt)
        bits{end+1} = sprintf('raw_dt=%gus', plan.rawDt * 1e6);
    end
    u = plan.responseUnits; if isempty(u); u = 'UNDECLARED'; end
    r = sprintf('response %s', u);
    if plan.responseScale ~= 1.0
        r = sprintf('%s x%g->canonical', r, plan.responseScale);
    end
    if plan.rectify; r = [r ', rectified']; end
    bits{end+1} = r;
    s = strjoin(bits, '; ');
end
