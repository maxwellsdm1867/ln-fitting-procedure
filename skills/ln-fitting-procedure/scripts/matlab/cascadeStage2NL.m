function [q, fv] = cascadeStage2NL(x, resp, nRestarts)
%CASCADESTAGE2NL  Stage 2: the 4-parameter nonlinearity, filter fixed.
%
% Prefits to the BINNED nonlinearity (CascadeGraph's sampleNl idea) before touching the raw
% cloud. The binned objective has no local minima to speak of, so the starting point barely
% matters, and it leaves gamma at a real value -- fminsearch perturbs a zero-valued coordinate
% by only 2.5e-4 absolute, so a gamma initialised at exactly 0 never moves. A DC-free filter
% convolved with a mean-subtracted stimulus has EXACTLY zero mean output, so the obvious
% gamma = -beta*mean(x) rule lands on precisely that trap.
PENALTY = 1e12;
xf = x(:); yf = resp(:);
s = sign(cascadeCorr(xf, yf)); if s == 0, s = 1; end
seed = [s*(max(yf)-min(yf)), 2.0/max(std(xf), eps), 0, 0];
if s < 0, seed(4) = max(yf); else, seed(4) = min(yf); end
seed(3) = -seed(2) * median(xf);

nBins = 30; [~, ord] = sort(xf); edges = round(linspace(0, numel(xf), nBins+1));
xb = zeros(nBins,1); yb = zeros(nBins,1);
for b = 1:nBins
    idx = ord(edges(b)+1:edges(b+1));
    xb(b) = mean(xf(idx)); yb(b) = mean(yf(idx));
end

qPre = cascadeNmRestarts(@(q) nlLoss(q, xb, yb, PENALTY), seed, 5);
[q, fv] = cascadeNmRestarts(@(q) nlLoss(q, xf, yf, PENALTY), qPre, nRestarts);
if abs(q(2)) < 1e-2 * abs(qPre(2))     % sigmoid collapsed into its linear regime; retry
    [q2, f2] = cascadeNmRestarts(@(q) nlLoss(q, xf, yf, PENALTY), qPre, nRestarts);
    if f2 < fv, q = q2; fv = f2; end
end
end

function e = nlLoss(q, x, y, PENALTY)
pred = q(1) * cascadeNormcdf(q(2)*x + q(3)) + q(4);
e = sum((y(:) - pred(:)).^2);
if ~isfinite(e), e = PENALTY; end
end
