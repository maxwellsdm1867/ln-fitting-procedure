function [best, bestF, exitflag] = cascadeNmRestarts(loss, v0, nRestarts)
%CASCADENMRESTARTS  Nelder-Mead with restarts.
%
% A "restart" means calling fminsearch again from the previous best. The simplex collapses as
% it converges, so restarting rebuilds a full-size one around the current point -- which is
% why ten cheap restarts explore more than one run with tight tolerances.
%
% Returns the exitflag of the final call: 0 means MaxFunEvals was exhausted, i.e. the
% optimizer did not converge and is disclaiming its own answer.
opts = optimset('TolX', 1e-4, 'TolFun', 1e-4, 'MaxFunEvals', 200*numel(v0), ...
                'MaxIter', 200*numel(v0), 'Display', 'off');
best = v0(:)'; bestF = loss(best); exitflag = [];
for k = 1:nRestarts
    [v, f, ef] = fminsearch(loss, best, opts);
    exitflag = ef;
    if isfinite(f) && f < bestF, best = v(:)'; bestF = f; end
end
end
