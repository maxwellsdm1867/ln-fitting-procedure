function [pFilt, losses, exitflag] = cascadeStage1Filter(stim, resp, dt, nRandomInits, nRestarts)
%CASCADESTAGE1FILTER  Stage 1: filter only, 6 params, prediction = scFact * conv(stim, filter).
%
% This stage is where fits are won or lost. The phase term makes the surface periodic, so a
% start on the wrong lobe never escapes -- which is why the default start alone is not enough
% and 19 further starts are drawn from the documented ranges.
PENALTY = 1e12;
nPts = size(stim, 2);
RANGES = [1 10; 0.005 0.1; 0.005 0.2; 0.01 0.1; -180 180; -500 500];
DEFAULT = [4, 0.02, 0.01, 0.02, 1, -100];

loss = @(v) localLoss(v, stim, resp, dt, nPts, PENALTY);
starts = [DEFAULT; bsxfun(@plus, RANGES(:,1)', ...
          bsxfun(@times, rand(nRandomInits-1, 6), diff(RANGES,1,2)'))];

bestV = []; bestF = inf; losses = zeros(size(starts,1),1); exitflag = [];
for i = 1:size(starts,1)
    [v, f, ef] = cascadeNmRestarts(loss, starts(i,:), nRestarts);
    losses(i) = f;
    if f < bestF, bestV = v; bestF = f; exitflag = ef; end
end
pFilt = bestV(1:5);
end

function e = localLoss(v, stim, resp, dt, nPts, PENALTY)
f = cascadeFilterSafe(v(1:5), nPts, dt);
if isempty(f), e = PENALTY; return; end
pred = v(6) * cascadeConv(stim, f);
e = sum((resp(:) - pred(:)).^2);
if ~isfinite(e), e = PENALTY; end
end
