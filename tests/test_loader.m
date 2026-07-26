function test_loader()
%TEST_LOADER  Does cascadeLoadEpochs actually obey the contract?

    tmp = tempname; mkdir(tmp);
    pass = 0; fail = 0;
    nRaw = 6000; dtRaw = 1e-4;

    fprintf('--- the declaration drives the load ---\n');
    d = mk('plain', struct('stim', randn(3, nRaw), 'resp', 5*randn(3, nRaw) - 20), ...
           struct('sample_interval_s', dtRaw, 'response_units', 'mV'));
    [s, r, info] = cascadeLoadEpochs(d, 'verbose', false);
    check('loads a declared recording', isequal(size(s), [3 60]) && isequal(size(r), [3 60]), ...
          sprintf('%s, factor %d', mat2str(size(s)), info.decimationFactor));
    check('  rawDt came from the contract', info.rawDtFromMetadata, ...
          sprintf('%g us', info.rawDt*1e6));
    check('  stimulus mean-subtracted', max(abs(mean(s, 2))) < 1e-12, ...
          sprintf('max|mean| = %.2e', max(abs(mean(s, 2)))));

    fprintf('\n--- units are applied, not just classified ---\n');
    raw = 0.02 * randn(3, nRaw);
    dV = mk('volts', struct('stim', randn(3, nRaw), 'resp', raw), ...
            struct('sample_interval_s', dtRaw, 'response_units', 'V'));
    [~, rV, iV] = cascadeLoadEpochs(dV, 'verbose', false);
    dMv = mk('millivolts', struct('stim', randn(3, nRaw), 'resp', raw * 1000), ...
             struct('sample_interval_s', dtRaw, 'response_units', 'mV'));
    [~, rM, ~] = cascadeLoadEpochs(dMv, 'verbose', false);
    check('V is scaled to mV', iV.responseScale == 1000, sprintf('x%g', iV.responseScale));
    check('  and matches the same data in mV', max(abs(rV(:) - rM(:))) < 1e-9, ...
          sprintf('max|diff| = %.3e mV', max(abs(rV(:) - rM(:)))));

    dR = mk('rate', struct('stim', randn(3, nRaw), 'resp', double(randi([0 60], 3, nRaw))), ...
            struct('sample_interval_s', dtRaw, 'response_units', 'spikes/s'));
    [~, ~, iR] = cascadeLoadEpochs(dR, 'verbose', false);
    check('rate declares rectify', iR.rectify && strcmp(iR.responseKind, 'rate'), ...
          sprintf('kind=%s rectify=%d', iR.responseKind, iR.rectify));

    fprintf('\n--- layout and variable names come from the contract ---\n');
    dT = mk('tx', struct('stim', randn(nRaw, 3), 'resp', randn(nRaw, 3)), ...
            struct('sample_interval_s', dtRaw, 'response_units', 'mV', ...
                   'orientation', 'time_x_epochs'));
    [sT, ~, iT] = cascadeLoadEpochs(dT, 'verbose', false);
    check('time_x_epochs transposed on load', isequal(size(sT), [3 60]), ...
          sprintf('%s, %s', mat2str(size(sT)), iT.orientation));

    dN = mk('names', struct('lightStim', randn(3, nRaw), 'voltage', randn(3, nRaw)), ...
            struct('sample_interval_s', dtRaw, 'response_units', 'mV', ...
                   'stim_var', 'lightStim', 'resp_var', 'voltage'));
    [sN, ~, iN] = cascadeLoadEpochs(dN, 'verbose', false);
    check('rig variable names honoured', isequal(size(sN), [3 60]), ...
          sprintf('%s/%s', iN.stimVar, iN.respVar));

    fprintf('\n--- refusals ---\n');
    dU = mk('noUnits', struct('stim', randn(3, nRaw), 'resp', randn(3, nRaw)), ...
            struct('sample_interval_s', dtRaw));
    [t, m] = tryFail(@() cascadeLoadEpochs(dU, 'verbose', false));
    check('unanswered question blocks the load', t, m);

    dNo = mk('noMeta', struct('stim', randn(3, nRaw), 'resp', randn(3, nRaw)), []);
    [t, m] = tryFail(@() cascadeLoadEpochs(dNo, 'verbose', false));
    check('undeclared recording refused', t, m);

    dBad = mk('badDt', struct('stim', randn(3, nRaw), 'resp', randn(3, nRaw)), ...
              struct('sample_interval_s', dtRaw, 'response_units', 'mV'));
    [t, m] = tryFail(@() cascadeLoadEpochs(dBad, 'dt', 0.00015, 'verbose', false));
    check('non-integer decimation refused', t, m);

    fprintf('\n--- contract off: works, but says so ---\n');
    ws = warning('off', 'cascadeLoadEpochs:contractOff');
    [sOff, ~, iOff] = cascadeLoadEpochs(dNo, 'contract', 'off', 'rawDt', dtRaw, 'verbose', false);
    warning(ws);
    check('off + explicit rawDt loads', isequal(size(sOff), [3 60]), mat2str(size(sOff)));
    check('  and is flagged DISABLED', strcmp(iOff.contract, 'DISABLED'), iOff.contract);
    check('  with an unresolved note', ~isempty(iOff.unresolved), ...
          iOff.unresolved{1}(1:min(52, end)));

    fprintf('\n%d passed, %d failed\n', pass, fail);
    rmdir(tmp, 's');

    function pth = mk(name, vars, meta)
        dd = fullfile(tmp, name); mkdir(dd);
        pth = fullfile(dd, 'cell.mat');
        save(pth, '-struct', 'vars');
        if ~isempty(meta)
            fid = fopen(fullfile(dd, 'meta.json'), 'w');
            fprintf(fid, '%s', jsonencode(meta)); fclose(fid);
        end
    end

    function [threw, msg] = tryFail(fn)
        try
            fn(); threw = false; msg = 'did NOT fail';
        catch ME
            threw = true; parts = strsplit(ME.message, newline);
            msg = strtrim(parts{1});
            if numel(msg) > 72; msg = [msg(1:69) '...']; end
        end
    end

    function check(name, cond, detail)
        if cond; pass = pass + 1; v = 'PASS'; else; fail = fail + 1; v = 'FAIL'; end
        if numel(detail) > 74; detail = [detail(1:71) '...']; end
        fprintf('  %s  %-34s %s\n', v, name, detail);
    end
end
