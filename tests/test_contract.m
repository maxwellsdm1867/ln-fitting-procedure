function test_contract()
%TEST_CONTRACT  Does the recording contract accept, reject and ask correctly?

    tmp = tempname; mkdir(tmp);
    pass = 0; fail = 0;

    fprintf('--- ACCEPT ---\n');
    write('good', struct('sample_interval_s', 1e-4, 'response_units', 'mV'));
    [plan, rep] = cascadeRecordingContract(mpath('good'));
    check('clean recording accepted', rep.accepted, rep.summary);
    check('  scale 1.0, no rectify', plan.responseScale == 1.0 && ~plan.rectify, ...
          sprintf('scale=%g rectify=%d', plan.responseScale, plan.rectify));

    write('volts', struct('sample_interval_s', 1e-4, 'response_units', 'V'));
    plan = cascadeRecordingContract(mpath('volts'));
    check('V converts to mV', plan.responseScale == 1000, sprintf('x%g', plan.responseScale));

    write('rate', struct('sample_interval_s', 1e-4, 'response_units', 'spikes/s'));
    plan = cascadeRecordingContract(mpath('rate'));
    check('rate rectifies', plan.rectify && strcmp(plan.responseKind, 'rate'), ...
          sprintf('kind=%s rectify=%d', plan.responseKind, plan.rectify));

    write('tx', struct('sample_interval_s', 1e-4, 'response_units', 'mV', ...
                       'orientation', 'time_x_epochs'));
    plan = cascadeRecordingContract(mpath('tx'));
    check('time_x_epochs -> transpose', plan.transpose, plan.orientation);

    write('vars', struct('sample_interval_s', 1e-4, 'response_units', 'mV', ...
                         'stim_var', 'lightStim', 'resp_var', 'voltage'));
    plan = cascadeRecordingContract(mpath('vars'));
    check('arbitrary var names mapped', ...
          strcmp(plan.stimVar, 'lightStim') && strcmp(plan.respVar, 'voltage'), ...
          sprintf('%s/%s', plan.stimVar, plan.respVar));

    fprintf('\n--- REJECT ---\n');
    write('nodt', struct('response_units', 'mV'));
    [r, m] = tryReject('nodt'); check('missing sample_interval_s', r, m);

    write('msdt', struct('sample_interval_s', 100, 'response_units', 'mV'));
    [r, m] = tryReject('msdt'); check('dt in ms caught', r, m);

    write('badori', struct('sample_interval_s', 1e-4, 'response_units', 'mV', ...
                           'orientation', 'sideways'));
    [r, m] = tryReject('badori'); check('bad orientation', r, m);

    [r, m] = tryReject('doesnotexist'); check('missing meta.json', r, m);

    fprintf('\n--- ASK (not guess) ---\n');
    write('noun', struct('sample_interval_s', 1e-4));
    [~, rep] = cascadeRecordingContract(mpath('noun'));
    check('undeclared units -> question', ~rep.accepted && numel(rep.questions) == 1, ...
          sprintf('%d question(s)', numel(rep.questions)));
    check('  question is answerable', ~isempty(rep.questions(1).options), ...
          rep.questions(1).question);

    write('weird', struct('sample_interval_s', 1e-4, 'response_units', 'arbitrary_daq_counts'));
    [~, rep] = cascadeRecordingContract(mpath('weird'));
    check('unknown unit -> question', numel(rep.questions) == 1, ...
          rep.questions(1).question(1:min(66, end)));

    write('scaled', struct('sample_interval_s', 1e-4, ...
                           'response_units', 'arbitrary_daq_counts', 'response_scale', 0.0025));
    [plan, rep] = cascadeRecordingContract(mpath('scaled'));
    check('explicit scale settles scaling', plan.responseScale == 0.0025, ...
          sprintf('x%g', plan.responseScale));
    check('  but still asks about rectify', ...
          numel(rep.questions) == 1 && strcmp(rep.questions(1).field, 'rectify'), ...
          sprintf('%d q, field=%s', numel(rep.questions), rep.questions(1).field));

    fprintf('\n--- every problem at once, not just the first ---\n');
    write('multi', struct('sample_interval_s', 500, 'orientation', 'sideways'));
    [~, rep] = cascadeRecordingContract(mpath('multi'), 'strict', false);
    check('both rejects reported', numel(rep.rejects) == 2, ...
          sprintf('%d rejects', numel(rep.rejects)));

    fprintf('\n%d passed, %d failed\n', pass, fail);
    rmdir(tmp, 's');

    % ---- nested helpers (share the parent workspace) ----------------------
    function write(name, s)
        d = fullfile(tmp, name);
        if ~exist(d, 'dir'); mkdir(d); end
        fid = fopen(fullfile(d, 'meta.json'), 'w');
        fprintf(fid, '%s', jsonencode(s));
        fclose(fid);
    end

    function p = mpath(name)
        p = fullfile(tmp, name, 'meta.json');
    end

    function check(name, cond, detail)
        if cond; pass = pass + 1; v = 'PASS'; else; fail = fail + 1; v = 'FAIL'; end
        fprintf('  %s  %-34s %s\n', v, name, detail);
    end

    function [rejected, msg] = tryReject(name)
        try
            cascadeRecordingContract(mpath(name));
            rejected = false; msg = 'did NOT reject';
        catch ME
            rejected = true;
            m = strsplit(ME.message, newline);
            msg = strtrim(m{min(2, numel(m))});
        end
    end
end
