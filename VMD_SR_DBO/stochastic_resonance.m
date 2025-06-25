function y = stochastic_resonance(imf, params)
% SR四稳态系统响应。输入信号x（即最优imfs）通过四稳势系统系统在噪声D的作用下产生随机共振效应输出增强后的信号y
%1.原始信号 
%2.VMD分解得到若干IMF分量（imf1, imf2, ..., imfN）
%3.选择/重构IMF分量（比如挑选其中有用的几个分量相加，常记为imf_recon）
%4.将重构IMF作为SR的输入
%5.得到SR增强后的信号y
% params: [u,v,w,k,D]
u = params(1);
v = params(2); 
w = params(3); 
k = params(4);
D = params(5);
dt = 1/1000;%仿真步长
y = zeros(size(imf));%输出信号初始化
state = 0;%初始状态
for i=1:length(imf)
    % 四稳势分段力
    if state < -w
        force = k;
    elseif state > w
        force = -k;
    else
        force = (u^2*v^2*w^2)*state - (u^2*v^2 + v^2*w^2 + w^2*u^2)*state^3 + ...
                (u^2 + v^2 + w^2)*state^5 - state^7;
    end
    noise = sqrt(2*D*dt)*randn();
    state = state + dt*(force + imf(i)) + noise;%Euler–Maruyama积分，适合SDE（随机微分方程）数值解
    y(i) = state;
end
end