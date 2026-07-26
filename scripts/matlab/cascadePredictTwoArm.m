function pred = cascadePredictTwoArm(p, stim, dt)
%CASCADEPREDICTTWOARM  NL1( filter1(stim) + NL2( filter2(stim) ) ).
%
% This is CascadeGraph's TwoArmLnHyperNode topology: ONE LINEAR ARM and one nonlinear arm
% summed, then a nonlinearity -- not two symmetric LN arms. Arm 1 has no nonlinearity and no
% gain of its own, which is exactly why alpha2 is identifiable: ParamFilterNode normalises to
% unit peak, so arm 1 is a fixed reference the second arm is weighed against.
nPts = size(stim,2);
f1 = cascadeFilterSafe([p.numFilt1 p.tauR1 p.tauD1 p.tauP1 p.phi1], nPts, dt);
f2 = cascadeFilterSafe([p.numFilt2 p.tauR2 p.tauD2 p.tauP2 p.phi2], nPts, dt);
if isempty(f1) || isempty(f2), pred = []; return; end
eps2 = 0; if isfield(p,'epsilon2'), eps2 = p.epsilon2; end
arm2 = p.alpha2 * cascadeNormcdf(p.beta2 * cascadeConv(stim, f2) + p.gamma2) + eps2;
pred = p.alpha1 * cascadeNormcdf(p.beta1 * (cascadeConv(stim, f1) + arm2) + p.gamma1) + p.epsilon1;
end
