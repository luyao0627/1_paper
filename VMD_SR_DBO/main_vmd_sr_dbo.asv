clc; clear; close all;

%% ==== 1. 测试信号生成 ====
fs = 1000;
t = 0:1/fs:2-1/fs;
N = length(t);
f1 = 50; f2 = 120; f3 = 200;
signal_clean = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.3*sin(2*pi*f3*t);
noise = 0.5*randn(size(t));
signal_noisy = signal_clean + noise;

%% ==== 2. DBO优化参数 ====
pop_size = 20; max_iter = 30;           % 种群、迭代
dim = 8;                                % 8个优化参数,k模态数，α惩罚因子，τ时间步长
% [K, alpha, tau, u, v, w, k, D],参数边界设置
lb = [3,  500, 0,   0.5, 0.5, 0.5, 0.1, 0];
ub = [10, 3000, 1,  2.0, 2.0, 1.8, 0.5,  3];

%% ==== 3. 适应度函数句柄 ====
fitness_func = @(params) fitness_vmd_sr(params, signal_noisy, signal_clean);

%% ==== 4. DBO主循环 ====
[best_params, best_fit, curve] = DBO(fitness_func, lb, ub, dim, pop_size, max_iter);

fprintf("最优参数: K=%d, alpha=%.1f, tau=%.3f, u=%.3f, v=%.3f, w=%.3f, k=%.3f, D=%.3f\n", ...
    round(best_params(1)), best_params(2), best_params(3), ...
    best_params(4), best_params(5), best_params(6), best_params(7), best_params(8));
fprintf("最大互相关系数: %.4f\n", best_fit);

%% ==== 5. 用最优参数处理信号 ====
K_opt = round(best_params(1)); 
alpha_opt = best_params(2); 
tau_opt = best_params(3);
u_opt = best_params(4); 
v_opt = best_params(5); 
w_opt = best_params(6);
k_opt = best_params(7); 
D_opt = best_params(8);

[u_modes, ~, ~] = VMD(signal_noisy, alpha_opt, tau_opt, K_opt);

% ==== (A) 显示所有IMF分量波形 ====
figure('Name','所有IMF分量');
for i=1:K_opt
    subplot(K_opt,1,i);
    plot(t, u_modes(i,:), 'b');
    ylabel(['IMF' num2str(i)]);
    if i==1, title('VMD分解所有IMF分量'); end
    if i==K_opt, xlabel('时间 (s)'); end
end

% ==== (B) 计算并显示IMF复杂度指标 ====
PEs = zeros(K_opt,1); Kurts = zeros(K_opt,1);
for i=1:K_opt
    PEs(i) = permutation_entropy(u_modes(i,:),3,1);%排列熵
    Kurts(i) = abs(kurtosis(u_modes(i,:)));%峭度
end
complex_score = PEs ./ (Kurts+eps);   % “/”联合筛选

% ==== (C) 柱状图对比 ====
figure('Name','IMF复杂度指标对比');
subplot(1,3,1);
bar(PEs,'b'); title('IMF排列熵'); xlabel('IMF编号'); ylabel('PE');
subplot(1,3,2);
bar(Kurts,'r'); title('IMF峭度'); xlabel('IMF编号'); ylabel('Kurtosis');
subplot(1,3,3);
bar(complex_score,'g'); title('IMF排列熵/峭度'); xlabel('IMF编号'); ylabel('PE/Kurtosis');

% ==== (D) 标记被选中的IMF编号 ====
[~, best_imf_idx] = min(complex_score);
disp(['用于SR增强的最佳IMF编号为：' num2str(best_imf_idx)]);
best_imf = u_modes(best_imf_idx,:);
subplot(1,3,3);
hold on;
bar(best_imf_idx,complex_score(best_imf_idx),'FaceColor','m'); % 用紫色突出

% ==== (E) SR增强 ====
sr_params = [u_opt, v_opt, w_opt, k_opt, D_opt];
enhanced = stochastic_resonance(best_imf, sr_params);

%% ==== 6. 可视化 ====
figure;
subplot(3,1,1); plot(t, signal_noisy); title('Noisy Signal');
subplot(3,1,2); plot(t, best_imf); title('Best IMF (VMD, by PE/Kurt)');
subplot(3,1,3); plot(t, enhanced); hold on; plot(t, signal_clean, '--'); legend('Enhanced','Clean'); title('SR Enhanced vs Clean');

figure;
plot(curve,'LineWidth',2);xlabel('迭代');ylabel('最大互相关');title('DBO收敛曲线');