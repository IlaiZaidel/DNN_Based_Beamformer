function [x, t] = istft(stft, awin, swin, hop, nfft, fs)
L = size(stft, 2);          % determine the number of signal frames
wlen = length(swin);        % determine the length of the synthesis window
xlen = wlen + (L-1)*hop;    % estimate the length of the signal vector
x = zeros(1, xlen);         % preallocate the signal vector

% reconstruction of the whole spectrum
if rem(nfft, 2)             
    % odd nfft excludes Nyquist point
    X = [stft; conj(flipud(stft(2:end, :)))];
else                        
    % even nfft includes Nyquist point
    X = [stft; conj(flipud(stft(2:end-1, :)))];
end

% columnwise IFFT on the STFT-matrix
xw = real(ifft(X));
xw = xw(1:wlen, :);

% Weighted-OLA
for l = 1:L
    x(1+(l-1)*hop : wlen+(l-1)*hop) = x(1+(l-1)*hop : wlen+(l-1)*hop) + ...
                                      (xw(:, l).*swin)';
end

% scaling of the signal
W0 = sum(awin.*swin);                  
x = x.*hop/W0;                      

% generation of the time vector
t = (0:xlen-1)/fs;                 

end