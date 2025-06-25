function [u, u_hat, omega] = VMD(signal, alpha, tau, K)
% 变分模态分解（简化教学版）
T = length(signal);
f_hat = fftshift(fft(signal));
omega = zeros(K,1);
u_hat = zeros(K,T);
u = zeros(K,T);

for k=1:K
    omega(k) = (k-1)/K*0.5;
end

for k=1:K
    f = ((0:T-1)/T)-0.5;
    H = exp(-alpha*(f-omega(k)).^2);
    u_hat(k,:) = f_hat .* H;
    u(k,:) = real(ifft(ifftshift(u_hat(k,:))));
end
end