function [ok, report] = cascadePreflight(dataPath, varargin)
%CASCADEPREFLIGHT  Can this recording be fitted? Answer BEFORE spending a job on it.
%
%   [ok, report] = cascadePreflight('cell.mat')
%   [ok, report] = cascadePreflight('data/*/')          % a whole directory of recordings
%
% A fit is about ten minutes. Discovering at the end of one that the response was in volts, or
% that the array was stored transposed, costs the whole ten -- and the version where it does
% NOT error but quietly fits the wrong thing costs considerably more than that. This is the
% cheap check that runs first.
%
% Returns ok=false rather than throwing, because the caller usually wants to preflight several
% recordings and see every verdict, not stop at the first bad one.
%
% REPORT(i): source, ok, rejects, questions, summary.

    p = inputParser;
    p.addParameter('verbose', true, @islogical);
    p.parse(varargin{:});

    targets = expand(dataPath);
    assert(~isempty(targets), 'cascadePreflight:none', 'nothing matched %s', dataPath);

    report = struct('source', {}, 'ok', {}, 'rejects', {}, 'questions', {}, 'summary', {});
    for i = 1:numel(targets)
        t = targets{i};
        if exist(t, 'dir') == 7
            metaPath = fullfile(t, 'meta.json');
            dataPath = findRecording(t);
        else
            d = fileparts(t); if isempty(d); d = '.'; end
            metaPath = fullfile(d, 'meta.json');
            dataPath = t;
        end
        [plan, rep] = cascadeRecordingContract(metaPath, 'strict', false);

        % Validating the declaration is only half of it. A meta.json can be internally perfect
        % and still contradict the array it describes -- a recording stored (time x epochs)
        % while declaring the default epochs_x_time passes every check above and then dies in
        % cascadeLoadEpochs, which is exactly the ten-minute job this function exists to save.
        % So open the file. whos('-file') reads the header only, so this stays cheap.
        rejects = rep.rejects;
        rejects = [rejects, layoutRejects(dataPath, plan)]; %#ok<AGROW>

        ok = isempty(rejects) && isempty(rep.questions);
        report(end+1) = struct('source', t, 'ok', ok, ...
                               'rejects', {rejects}, 'questions', rep.questions, ...
                               'summary', rep.summary); %#ok<AGROW>
    end

    ok = all([report.ok]);

    if p.Results.verbose
        for i = 1:numel(report)
            r = report(i);
            if r.ok
                fprintf('OK    %s  [%s]\n', r.source, r.summary);
            else
                fprintf('BLOCK %s\n', r.source);
                for j = 1:numel(r.rejects)
                    fprintf('      reject: %s\n', r.rejects{j});
                end
                for j = 1:numel(r.questions)
                    fprintf('      ask:    %s\n', r.questions(j).question);
                end
            end
        end
        n = numel(report);
        fprintf('\n%d/%d ready to fit%s\n', sum([report.ok]), n, ...
                ternary(ok, '', ' -- resolve the above before submitting the job'));
    end
end

function p = findRecording(d)
%FINDRECORDING  The .mat in a recording directory, if there is exactly one obvious candidate.
    p = '';
    for name = {'cell.mat', 'data.mat'}
        c = fullfile(d, name{1});
        if exist(c, 'file') == 2; p = c; return; end
    end
    f = dir(fullfile(d, '*.mat'));
    if numel(f) == 1
        p = fullfile(f(1).folder, f(1).name);
    end
end

function rj = layoutRejects(dataPath, plan)
%LAYOUTREJECTS  Does the stored array actually agree with what the declaration says?
%
% Returns the reasons this recording would fail to load, so they arrive here rather than ten
% minutes into a job. Silent when the file cannot be inspected -- an unreadable file is the
% contract's problem to report, not this check's to guess at.
    rj = {};
    if isempty(dataPath) || exist(dataPath, 'file') ~= 2; return; end
    try
        w = whos('-file', dataPath);          % header only: no array is read
    catch
        return
    end
    if isempty(w); return; end

    names = {w.name};
    sv = plan.stimVar; rv = plan.respVar;
    iS = find(strcmp(names, sv), 1); iR = find(strcmp(names, rv), 1);
    if isempty(iS) || isempty(iR)
        missing = {};
        if isempty(iS); missing{end+1} = sv; end
        if isempty(iR); missing{end+1} = rv; end
        rj{end+1} = sprintf(['%s holds %s, but the contract names ''%s''. Declare stim_var / ' ...
            'resp_var, or the load will fail on a variable that is not there.'], ...
            dataPath, strjoin(names, ', '), strjoin(missing, ''' and '''));
        return
    end

    szS = w(iS).size; szR = w(iR).size;
    if ~isequal(szS, szR)
        rj{end+1} = sprintf('%s: %s is %s but %s is %s -- stimulus and response differ', ...
            dataPath, sv, mat2str(szS), rv, mat2str(szR));
        return
    end
    if numel(szS) ~= 2
        rj{end+1} = sprintf('%s: %s is %d-dimensional; expected an (epochs x time) matrix', ...
            dataPath, sv, numel(szS));
        return
    end

    stored = szS;
    if plan.transpose; szS = fliplr(szS); end
    if szS(1) > szS(2)
        applied = '';
        if plan.transpose
            applied = sprintf(', which transposes to %s', mat2str(szS));
        end
        rj{end+1} = sprintf(['%s is stored %s and the contract declares ''%s''%s -- that is ' ...
            'more epochs than time points, so the array and its declaration disagree. Declare ' ...
            'orientation as ''%s''.'], ...
            dataPath, mat2str(stored), plan.orientation, applied, ...
            ternary(plan.transpose, 'epochs_x_time', 'time_x_epochs'));
    end
end

function out = expand(pat)
    if exist(pat, 'file') == 2 || exist(pat, 'dir') == 7
        out = {pat}; return
    end
    d = dir(pat);
    out = {};
    for i = 1:numel(d)
        if strcmp(d(i).name, '.') || strcmp(d(i).name, '..'); continue; end
        out{end+1} = fullfile(d(i).folder, d(i).name); %#ok<AGROW>
    end
end

function out = ternary(c, a, b)
    if c; out = a; else; out = b; end
end
