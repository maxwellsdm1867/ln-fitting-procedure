function metaPath = cascadeWriteContract(dataPath, fields, varargin)
%CASCADEWRITECONTRACT  Write a confirmed contract to meta.json, beside the recording.
%
%   metaPath = cascadeWriteContract('cell.mat', fields)
%   metaPath = cascadeWriteContract('cell.mat', fields, 'overwrite', true)
%
% Closes the loop that cascadeInferContract opens: infer, confirm with the scientist, write it
% down. Writing it down is the part that matters -- an answer given in conversation is gone
% next session, and the loader would be back to guessing.
%
% The contract is validated BEFORE it lands. Writing an invalid meta.json would turn a
% question that was already answered into a rejection later, at fit time, which is the one
% place it costs a job.
%
% Existing fields are preserved: cell_type, protocol, notes and anything else the recording
% already carried are not the loader's business and are not for this function to drop.

    p = inputParser;
    p.addParameter('overwrite', false, @islogical);
    p.parse(varargin{:});

    assert(isstruct(fields), 'cascadeWrite:fields', 'FIELDS must be a struct');
    here = fileparts(dataPath);
    if isempty(here); here = '.'; end
    metaPath = fullfile(here, 'meta.json');

    existing = struct();
    if exist(metaPath, 'file') == 2
        try
            existing = jsondecode(fileread(metaPath));
        catch
            existing = struct();
        end
        if ~p.Results.overwrite
            clash = intersect(fieldnames(existing), fieldnames(fields));
            changed = {};
            for i = 1:numel(clash)
                a = existing.(clash{i}); b = fields.(clash{i});
                if ~isequal(a, b); changed{end+1} = clash{i}; end %#ok<AGROW>
            end
            assert(isempty(changed), 'cascadeWrite:wouldOverwrite', ...
                ['%s already declares %s differently. Pass ''overwrite'', true if the new ' ...
                 'answer is the right one -- silently rewriting a declaration the scientist ' ...
                 'made earlier is how a contract stops being trustworthy.'], ...
                metaPath, strjoin(changed, ', '));
        end
    end

    merged = existing;
    fn = fieldnames(fields);
    for i = 1:numel(fn)
        v = fields.(fn{i});
        if isempty(v) && ~isnumeric(v)
            continue                       % never write a blank string as a declaration
        end
        merged.(fn{i}) = v;
    end

    % Validate before landing, using a scratch copy so a rejected contract never reaches disk.
    scratch = [tempname '.json'];
    fid = fopen(scratch, 'w');
    assert(fid > 0, 'cascadeWrite:tmp', 'could not open a scratch file');
    fprintf(fid, '%s', jsonencode(merged));
    fclose(fid);
    cleanup = onCleanup(@() delete(scratch));

    [~, report] = cascadeRecordingContract(scratch, 'strict', false);
    assert(isempty(report.rejects), 'cascadeWrite:invalid', ...
        'refusing to write an invalid contract to %s:\n%s', metaPath, ...
        strjoin(cellfun(@(s) ['  - ' s], report.rejects, 'UniformOutput', false), newline));

    fid = fopen(metaPath, 'w');
    assert(fid > 0, 'cascadeWrite:open', 'could not write %s', metaPath);
    fprintf(fid, '%s\n', jsonencode(merged, 'PrettyPrint', true));
    fclose(fid);

    if ~isempty(report.questions)
        fprintf('[cascadeContract] wrote %s, still unresolved: %s\n', metaPath, ...
                strjoin({report.questions.field}, ', '));
    else
        fprintf('[cascadeContract] wrote %s -- %s\n', metaPath, report.summary);
    end
end
