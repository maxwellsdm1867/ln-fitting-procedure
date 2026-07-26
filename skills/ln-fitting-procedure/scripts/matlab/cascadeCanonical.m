function p = cascadeCanonical(p, dt)
%CASCADECANONICAL  Resolve the four exact degeneracies in this parameterization.
%
% All four leave the prediction bit-identical, and a fit lands on either branch at random, so
% alpha, gamma, epsilon, tauP and the time constants are not comparable across fits until a
% branch is chosen. Resolving them is bookkeeping, but unresolved bookkeeping masquerades as
% science: a recovery test on this model once reported 205% error on tauR, which reads as
% "not identifiable" and was purely an unresolved sign.
%
%   1. tauR, tauD signs   -- the filter takes abs() of both, so their signs are unconstrained
%   2. tauP sign          -- cos is even, so negating tauP and phi together is a no-op
%   3. tauP aliasing      -- the filter is sampled at t = k*dt, so dt/tauP and 1-dt/tauP alias
%   4. overall sign       -- (phi+180, alpha->-alpha, gamma->-gamma, epsilon->alpha+epsilon)
%
% Applied in that order: de-aliasing assumes a positive period. a_fb flips with alpha when
% present, because the loop gain -- not a_fb alone -- is what is interpretable.

p.tauR = abs(p.tauR);
p.tauD = abs(p.tauD);
if isfield(p,'tauR2'), p.tauR2 = abs(p.tauR2); p.tauD2 = abs(p.tauD2); end

if p.tauP < 0, p.tauP = -p.tauP; p.phi = -p.phi; end
nu = dt / p.tauP;
if nu > 0.5, p.tauP = dt / (1 - nu); p.phi = -p.phi; end

if p.alpha < 0
    p.phi     = mod(p.phi + 360, 360) - 180;
    p.epsilon = p.alpha + p.epsilon;
    p.alpha   = -p.alpha;
    p.gamma   = -p.gamma;
    if isfield(p,'a_fb'), p.a_fb = -p.a_fb; end
end
p.phi = mod(p.phi + 180, 360) - 180;
end
