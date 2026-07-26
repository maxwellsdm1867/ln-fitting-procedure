function y = cascadeNormcdf(z)
%CASCADENORMCDF  Standard normal CDF without the Statistics Toolbox.
% Identical to normcdf(z,0,1) (verified max difference exactly 0).
y = 0.5 * erfc(-z ./ sqrt(2));
end
