function pred = cascadeFreeRun(filtered, alpha, beta, gamma, epsilon, aFb, tauFb, nFbBins, dt)
%CASCADEFREERUN  Free-running GLM prediction: the feedback sees the MODEL's own output.
%
%   pred(t) = NL( filtered(t) + sum_{k=1..K} h(k)*pred(t-k) ),  h(k) = aFb*exp(-k*dt/tauFb)
%
% Never teacher-forced. With the observed response available one lag back, the kernel's
% cheapest strategy is to copy it -- any smooth response predicts its own recent past well --
% so the fit converges to a near-identity autoregression, reports a high R^2, and has learned
% nothing about how the stimulus drives the cell.
%
% The sum is evaluated by recursion rather than by a dot product, which is exact because the
% kernel is exponential. With rho = exp(-dt/tauFb) and S(t) = sum_k rho^k pred(t-k),
%
%     S(t+1) = rho*(pred(t) + S(t)) - rho^(K+1) * pred(t-K)
%
% so each step is O(1) instead of O(K), and nothing is reallocated. This is the whole cost of
% GLM fitting -- the loop runs once per objective evaluation, and stage 2 alone is 5 starts x
% 10 restarts x 200*6 evaluations -- so the difference is minutes versus hours.
tauFb = max(tauFb, dt);
rho   = exp(-dt / tauFb);
rhoK1 = rho^(nFbBins + 1);
[nEp, T] = size(filtered);
pred = zeros(nEp, T);
c    = 1/sqrt(2);

for e = 1:nEp
    buf = zeros(nFbBins, 1);     % buf(idx) is the oldest retained prediction, pred(t-K)
    idx = 1;
    S   = 0;
    fe  = filtered(e, :);
    for t = 1:T
        y = alpha * (0.5 * erfc(-(beta*(fe(t) + aFb*S) + gamma) * c)) + epsilon;
        oldest = buf(idx);
        S = rho*(y + S) - rhoK1*oldest;
        buf(idx) = y;
        idx = idx + 1; if idx > nFbBins, idx = 1; end
        pred(e, t) = y;
    end
end
end
