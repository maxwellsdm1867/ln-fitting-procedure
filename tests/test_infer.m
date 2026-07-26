function test_infer()
%TEST_INFER  Does contract inference propose the right things, and admit what it cannot know?

    tmp = tempname; mkdir(tmp);
    pass = 0; fail = 0;

    stim = randn(3, 6000); resp = 5 * randn(3, 6000) - 20;

    fprintf('--- names ---\n');
    p1 = fullfile(tmp, 'a.mat'); save(p1, 'stim', 'resp');
    pr = cascadeInferContract(p1);
    check('canonical names certain', ...
          strcmp(pr.fields.stim_var, 'stim') && strcmp(pr.confidence.stim_var, 'certain'), ...
          sprintf('%s/%s', pr.fields.stim_var, pr.fields.resp_var));

    lightStim = stim; voltage = resp;
    p2 = fullfile(tmp, 'b.mat'); save(p2, 'lightStim', 'voltage');
    pr = cascadeInferContract(p2);
    check('rig names mapped', ...
          strcmp(pr.fields.stim_var, 'lightStim') && strcmp(pr.fields.resp_var, 'voltage'), ...
          sprintf('%s/%s', pr.fields.stim_var, pr.fields.resp_var));

    fprintf('\n--- layout ---\n');
    check('wide array -> epochs_x_time', strcmp(pr.fields.orientation, 'epochs_x_time'), ...
          pr.evidence.orientation);

    stimT = stim.'; respT = resp.';
    p3 = fullfile(tmp, 'c.mat'); save(p3, 'stimT', 'respT');
    pr3 = cascadeInferContract(p3);
    check('tall array -> time_x_epochs', strcmp(pr3.fields.orientation, 'time_x_epochs'), ...
          pr3.evidence.orientation);

    sq = randn(4, 4); sq2 = randn(4, 4);
    p4 = fullfile(tmp, 'd.mat'); stim = sq; resp = sq2; save(p4, 'stim', 'resp');
    pr4 = cascadeInferContract(p4);
    check('square array -> must_ask', strcmp(pr4.confidence.orientation, 'must_ask'), ...
          pr4.evidence.orientation);

    fprintf('\n--- sampling interval: the one nothing can reveal ---\n');
    check('absent -> must_ask', strcmp(pr.confidence.sample_interval_s, 'must_ask'), ...
          pr.evidence.sample_interval_s);

    stim = randn(3, 6000); resp = randn(3, 6000); sample_interval_s = 1e-4;
    p5 = fullfile(tmp, 'e.mat'); save(p5, 'stim', 'resp', 'sample_interval_s');
    pr5 = cascadeInferContract(p5);
    check('present -> likely, value read', ...
          strcmp(pr5.confidence.sample_interval_s, 'likely') && ...
          abs(pr5.fields.sample_interval_s - 1e-4) < 1e-12, pr5.evidence.sample_interval_s);

    Fs = 10000;
    p6 = fullfile(tmp, 'f.mat'); save(p6, 'stim', 'resp', 'Fs');
    pr6 = cascadeInferContract(p6);
    check('a RATE is inverted, not taken raw', ...
          abs(pr6.fields.sample_interval_s - 1e-4) < 1e-12, pr6.evidence.sample_interval_s);

    fprintf('\n--- units: proposed, never settled ---\n');
    check('units always must_ask', strcmp(pr.confidence.response_units, 'must_ask'), ...
          pr.evidence.response_units);

    stim = randn(3, 5000); resp = double(randi([0 60], 3, 5000));
    p7 = fullfile(tmp, 'g.mat'); save(p7, 'stim', 'resp');
    pr7 = cascadeInferContract(p7);
    check('non-neg integers -> spikes/s proposed', strcmp(pr7.fields.response_units, 'spikes/s'), ...
          pr7.evidence.response_units);

    resp = 0.02 * randn(3, 5000);
    p8 = fullfile(tmp, 'h.mat'); save(p8, 'stim', 'resp');
    pr8 = cascadeInferContract(p8);
    check('sub-1 span -> V proposed (1000x trap)', strcmp(pr8.fields.response_units, 'V'), ...
          pr8.evidence.response_units);

    fprintf('\n--- mustAsk is the checklist the agent has to clear ---\n');
    check('mustAsk non-empty on a bare file', numel(pr.mustAsk) >= 2, ...
          strjoin(pr.mustAsk, ', '));

    fprintf('\n%d passed, %d failed\n', pass, fail);
    rmdir(tmp, 's');

    function check(name, cond, detail)
        if cond; pass = pass + 1; v = 'PASS'; else; fail = fail + 1; v = 'FAIL'; end
        if numel(detail) > 78; detail = [detail(1:75) '...']; end
        fprintf('  %s  %-36s %s\n', v, name, detail);
    end
end
