function out = cascadeVerifyConv(convFn, nPoints, verbose)
%CASCADEVERIFYCONV  Check a convolution against the three conventions, no CascadeGraph needed.
%
%   cascadeVerifyConv(@cascadeConv)                  % the reference: all pass
%   cascadeVerifyConv(@(s,f) myConv(s,f))            % your own implementation
%
% Point this at any convolution -- a helper in an existing script, a port, anything you did
% not get from this module -- and it reports WHICH convention is wrong rather than just that
% the numbers differ. Each property is probed with a stimulus chosen so a correct and an
% incorrect implementation differ qualitatively, not in the sixth decimal.
%
%   1. CIRCULAR, not time-domain causal. The end of an epoch's stimulus is the history for
%      its first bins. An impulse in the last sample must produce a response that wraps to
%      the start; a causal or zero-padded convolution leaves those bins empty.
%   2. FILTER AT FULL STIMULUS LENGTH, not truncated. A deliberate tap at a long lag must
%      contribute; a truncated kernel silently drops it.
%   3. PER-EPOCH for (epochs x time) data. With epoch 1 driven and epoch 2 silent, epoch 2
%      must be exactly zero; flattening splices one trial onto the next and leaks.

if nargin < 2 || isempty(nPoints), nPoints = 256; end
if nargin < 3 || isempty(verbose), verbose = true; end
n = nPoints;
names = {'circular_not_causal', 'filter_full_length', 'per_epoch'};
passed = false(1,3); detail = cell(1,3);

% 1 -- circular, not causal
f = zeros(n,1); f(2) = 1; f(3) = 0.5; f(4) = 0.25;
stim = zeros(1,n); stim(end) = 1;
y = convFn(stim, f);
wrapped = max(abs(y(1,1:3)));
passed(1) = wrapped > 1e-9;
detail{1} = sprintf(['impulse at the last sample produced |response| = %.3g in the first 3 ' ...
    'bins; circular wraps (expect ~1), causal or zero-padded does not (expect 0)'], wrapped);

% 2 -- filter used at full length
far = floor(n/2);
f2 = zeros(n,1); f2(2) = 1; f2(far) = 1;
stim2 = zeros(1,n); stim2(1) = 1;
y2 = convFn(stim2, f2);
gotFar = abs(y2(1, far));
passed(2) = gotFar > 1e-9;
detail{2} = sprintf(['a filter tap at lag %d contributed %.3g; a truncated kernel drops it ' ...
    '(expect 0), full length keeps it (expect ~1)'], far-1, gotFar);

% 3 -- per-epoch, no leakage between trials
f3 = zeros(n,1); f3(1:8) = 1;
stim3 = zeros(2,n); stim3(1,:) = 1;
y3 = convFn(stim3, f3);
leak = max(abs(y3(2,:)));
passed(3) = isequal(size(y3), [2 n]) && leak < 1e-9;
detail{3} = sprintf(['with epoch 1 driven and epoch 2 silent, epoch 2 had max |value| = ' ...
    '%.3g and size %s; per-epoch convolution leaves it exactly 0'], leak, mat2str(size(y3)));

if verbose
    for k = 1:3
        if passed(k), tag = 'PASS'; else, tag = 'FAIL'; end
        fprintf('  %s  %-22s %s\n', tag, names{k}, detail{k});
    end
    if all(passed)
        fprintf('  -> convolution matches the convention\n');
    else
        fprintf('  -> CONVENTION VIOLATED\n');
    end
end
out = struct('ok', all(passed), 'names', {names}, 'passed', passed, 'detail', {detail});
end
