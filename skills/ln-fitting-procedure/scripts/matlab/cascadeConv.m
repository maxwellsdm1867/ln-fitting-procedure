function g = cascadeConv(stim, filt)
%CASCADECONV  Circular FFT convolution, per epoch, filter at full stimulus length.
% Matches ParamFilterNode.processTempParams. The wrap-around is deliberate: padding changes
% the model and breaks comparability with reference fits.
g = real(ifft(fft(stim') .* fft(filt)))';
end
