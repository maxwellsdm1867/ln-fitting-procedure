function f = cascadeFilterSafe(v, nPts, dt)
%CASCADEFILTERSAFE  ParamFilterNode filter, or [] if the parameters are unusable.
% tauR and tauD are wrapped in abs() so an optimizer walking them negative is a no-op rather
% than an overflow; the filter is identical either way.
f = [];
if ~all(isfinite(v)) || v(4) == 0, return; end
s = struct('numFilt', v(1), 'tauR', abs(v(2)), 'tauD', abs(v(3)), 'tauP', v(4), 'phi', v(5));
try
    f = ParamFilterNode.getFilterWithParams(s, nPts, dt);
catch
    f = []; return;
end
if ~all(isfinite(f)), f = []; end
end
