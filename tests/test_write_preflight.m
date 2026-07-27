function test_write_preflight()
%TEST_WRITE_PREFLIGHT  infer -> confirm -> write, and the gate that runs before a job.

    tmp = tempname; mkdir(tmp);
    pass = 0; fail = 0;
    stim = randn(3, 6000); resp = 5 * randn(3, 6000) - 20; %#ok<NASGU>

    cellDir = fullfile(tmp, 'cellA'); mkdir(cellDir);
    dataPath = fullfile(cellDir, 'cell.mat');
    save(dataPath, 'stim', 'resp');

    fprintf('--- preflight BLOCKS an undeclared recording ---\n');
    [ok, rep] = cascadePreflight(dataPath, 'verbose', false);
    check('undeclared -> not ok', ~ok, sprintf('%d reject(s)', numel(rep(1).rejects)));
    check('  reject names the missing meta', ...
          any(contains(rep(1).rejects, 'meta.json')), rep(1).rejects{1}(1:min(60,end)));

    fprintf('\n--- infer proposes, we confirm, write lands ---\n');
    pr = cascadeInferContract(dataPath);
    f = pr.fields;
    check('infer flags what must be asked', numel(pr.mustAsk) >= 2, strjoin(pr.mustAsk, ', '));
    % Stand in for the scientist answering the AskUserQuestion prompts.
    f.sample_interval_s = 1e-4;
    f.response_units = 'mV';
    f.stimulus_units = 'arbitrary (contrast)';
    mp = cascadeWriteContract(dataPath, f);
    check('meta.json written', exist(mp, 'file') == 2, mp);

    fprintf('\n--- preflight now PASSES ---\n');
    [ok, rep] = cascadePreflight(dataPath, 'verbose', false);
    check('declared -> ok', ok, rep(1).summary);

    fprintf('\n--- an invalid contract never reaches disk ---\n');
    bad = fullfile(tmp, 'cellB'); mkdir(bad);
    dp2 = fullfile(bad, 'cell.mat'); save(dp2, 'stim', 'resp');
    threw = false; msg = '';
    try
        cascadeWriteContract(dp2, struct('sample_interval_s', 500, 'response_units', 'mV'));
    catch ME
        threw = true; m = strsplit(ME.message, newline); msg = strtrim(m{min(2, numel(m))});
    end
    check('ms-valued interval refused', threw, msg(1:min(70, end)));
    check('  and no file was left behind', exist(fullfile(bad, 'meta.json'), 'file') ~= 2, '');

    fprintf('\n--- existing declarations are not silently rewritten ---\n');
    threw = false;
    try
        cascadeWriteContract(dataPath, struct('response_units', 'pA'));
    catch
        threw = true;
    end
    check('clashing rewrite refused', threw, 'needs overwrite=true');
    cascadeWriteContract(dataPath, struct('response_units', 'pA'), 'overwrite', true);
    m2 = jsondecode(fileread(mp));
    check('overwrite=true honoured', strcmp(m2.response_units, 'pA'), m2.response_units);
    check('unrelated fields preserved', isfield(m2, 'stimulus_units'), ...
          'stimulus_units survived');

    fprintf('\n--- preflight over a directory of recordings ---\n');
    batch = fullfile(tmp, 'batch'); mkdir(batch);
    for k = 1:3
        d = fullfile(batch, sprintf('r%d', k)); mkdir(d);
        save(fullfile(d, 'cell.mat'), 'stim', 'resp');
        if k < 3
            cascadeWriteContract(fullfile(d, 'cell.mat'), ...
                struct('sample_interval_s', 1e-4, 'response_units', 'mV'));
        end
    end
    [ok, rep] = cascadePreflight(fullfile(batch, '*'), 'verbose', false);
    check('mixed batch -> not ok', ~ok, sprintf('%d/%d ready', sum([rep.ok]), numel(rep)));
    check('  the bad one is identified', sum([rep.ok]) == 2 && numel(rep) == 3, ...
          'two declared, one not');

    fprintf('\n--- the declaration is checked AGAINST the array, not just for itself ---\n');
    % A meta.json can be internally perfect and still contradict its own recording. Validating
    % only the declaration passes this and then dies ten minutes later in cascadeLoadEpochs,
    % which is the job preflight exists to save.
    tx = fullfile(tmp, 'tx'); mkdir(tx);
    txPath = fullfile(tx, 'cell.mat');
    stimT = stim.'; respT = resp.';                       %#ok<NASGU> stored (time x epochs)
    save(txPath, 'stimT', 'respT');
    cascadeWriteContract(txPath, struct('sample_interval_s', 1e-4, 'response_units', 'mV', ...
                                        'stim_var', 'stimT', 'resp_var', 'respT'));
    [ok, rep] = cascadePreflight(txPath, 'verbose', false);
    check('stored transpose blocked', ~ok, sprintf('%d reject(s)', numel(rep(1).rejects)));
    check('  reject names the layout', ...
          ~isempty(rep(1).rejects) && contains(rep(1).rejects{1}, 'time_x_epochs'), ...
          rep(1).rejects{1}(1:min(64, end)));

    cascadeWriteContract(txPath, struct('orientation', 'time_x_epochs'), 'overwrite', true);
    [ok, ~] = cascadePreflight(txPath, 'verbose', false);
    check('  declaring it clears the gate', ok, 'orientation time_x_epochs');

    nv = fullfile(tmp, 'novar'); mkdir(nv);
    nvPath = fullfile(nv, 'cell.mat');
    save(nvPath, 'stim', 'resp');
    cascadeWriteContract(nvPath, struct('sample_interval_s', 1e-4, 'response_units', 'mV', ...
                                        'stim_var', 'lightStim'));
    [ok, rep] = cascadePreflight(nvPath, 'verbose', false);
    check('declared variable that is absent blocked', ~ok, ...
          sprintf('%d reject(s)', numel(rep(1).rejects)));

    good = fullfile(tmp, 'good2'); mkdir(good);
    gPath = fullfile(good, 'cell.mat');
    save(gPath, 'stim', 'resp');
    cascadeWriteContract(gPath, struct('sample_interval_s', 1e-4, 'response_units', 'mV'));
    [ok, ~] = cascadePreflight(gPath, 'verbose', false);
    check('a sound recording still passes', ok, 'no false positive from the shape check');

    fprintf('\n%d passed, %d failed\n', pass, fail);
    rmdir(tmp, 's');

    function check(name, cond, detail)
        if cond; pass = pass + 1; v = 'PASS'; else; fail = fail + 1; v = 'FAIL'; end
        if numel(detail) > 74; detail = [detail(1:71) '...']; end
        fprintf('  %s  %-34s %s\n', v, name, detail);
    end
end
