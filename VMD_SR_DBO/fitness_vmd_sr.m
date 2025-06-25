function fit = fitness_vmd_sr(params, signal_noisy, signal_clean)
% 适应度: VMD+SR全参数优化, IMF用(排列熵/峭度)联合筛选, 互相关为优化目标
% params: [K, alpha, tau, u, v, w, k, D]
% signal_noisy: 输入带噪信号
% signal_clean: 理想信号
K = round(params(1));% VMD模态数（取整）
alpha = params(2);% VMD惩罚参数
tau = params(3);% VMD噪声容忍度
u = params(4);% 四稳势函数参数u
v = params(5);% 四稳势函数参数v
w = params(6);% 四稳势函数参数w
k = params(7);% 四稳势函数参数k
D = params(8);% SR噪声强度

try
    [u_modes, ~, ~] = VMD(signal_noisy, alpha, tau, K);%对含噪信号进行VMD分解得到K个IMF分量（u_modes）

    % 用(排列熵/峭度)复杂度指标联合筛选最佳IMF
    PEs = zeros(K,1); Kurts = zeros(K,1);
    for i=1:K
        PEs(i) = permutation_entropy(u_modes(i,:),3,1);%排列熵，衡量信号复杂度，值越大表示越无序（噪声成分多）
        Kurts(i) = abs(kurtosis(u_modes(i,:)));%峭度绝对值，衡量信号的尖峰特性，绝对值越大表示含有更多冲击成分
    end
    complex_score = PEs./(Kurts+eps); 
    % 复杂度得分，比值小：PE小（有序）且Kurt大（冲击特征明显）→ 有用信号
                %比值大：PE大（无序）且Kurt小（平坦）→ 噪声成分
    [~, best_imf_idx] = min(complex_score); % 找最小得分的IMF
    imf = u_modes(best_imf_idx,:);
    
    % SR增强，对选出的最佳IMF进行本文SR系统增强
    sr_params = [u,v,w,k,D];
    enhanced = stochastic_resonance(imf, sr_params);

    % 适应度 = 增强后与理想信号最大互相关
    fit = max(xcorr(enhanced, signal_clean, 'coeff'));
catch
    fit = -1e3; % 异常惩罚
end
end