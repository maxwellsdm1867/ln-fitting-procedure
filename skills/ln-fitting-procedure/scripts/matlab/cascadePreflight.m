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
        else
            d = fileparts(t); if isempty(d); d = '.'; end
            metaPath = fullfile(d, 'meta.json');
        end
        [~, rep] = cascadeRecordingContract(metaPath, 'strict', false);
        report(end+1) = struct('source', t, 'ok', rep.accepted, ...
                               'rejects', {rep.rejects}, 'questions', rep.questions, ...
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
