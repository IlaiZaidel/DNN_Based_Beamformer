
function [STFT, f, t] = stft(x, win, hop, nfft)
fs=16000;
x = x(:); 
xlen = length(x);
wlen = length(win);
NUP = ceil((nfft+1)/2);    
L = 1+fix((xlen-wlen)/hop);
STFT = zeros(NUP,L);
for l = 0:L-1
    x_w = x(1+l*hop : wlen+l*hop).*win;
    X = fft(x_w, nfft);
    STFT(:, 1+l) = X(1:NUP);
end
t = (wlen/2:hop:wlen/2+(L-1)*hop)/fs;
f = (0:NUP-1)*fs/nfft;
end