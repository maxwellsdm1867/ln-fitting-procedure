function r = cascadeCorr(a, b)
%CASCADECORR  Pearson correlation of two vectors, base MATLAB only.
%
% Deliberately not corr(): that lives in the Statistics and Machine Learning Toolbox, and
% nothing else in this fitting layer needs a toolbox. A fitting pipeline that silently
% requires a licence is a pipeline that does not run on a colleague's machine.
a = a(:) - mean(a(:));
b = b(:) - mean(b(:));
d = sqrt(sum(a.^2) * sum(b.^2));
if d == 0, r = 0; else, r = sum(a .* b) / d; end
end
