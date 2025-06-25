clc; clear; close all;

%% ==== 1. 基本参数 ====
% 生成测试信号
fs = 1000; t = 0:1/fs:2-1/fs;
f1 = 50; f2 = 120; f3 = 200;
signal_clean = sin(2*pi*f1*t) + 0.5*sin(2*pi*f2*t) + 0.3*sin(2*pi*f3*t);
noise = 0.5*randn(size(t));
signal_noisy = signal_clean + noise;

pop_size = 20;      % 三种算法都用这个种群/粒子数
max_iter = 30;      % 三种算法都用这个最大迭代次数
dim = 8;
lb = [3,500,0,0.5,0.5,0.5,0.05,0];
ub = [10,3000,1,2.0,2.0,2.0,0.5,0.2];

fitness_func = @(params) fitness_vmd_sr(params, signal_noisy, signal_clean);

N_rep = 10; % 重复实验次数

% 记录所有实验的结果
all_fit_DBO = zeros(N_rep,1); all_time_DBO = zeros(N_rep,1);
all_fit_PSO = zeros(N_rep,1); all_time_PSO = zeros(N_rep,1);
all_fit_GA  = zeros(N_rep,1); all_time_GA  = zeros(N_rep,1);

all_curve_DBO = zeros(N_rep, max_iter);
all_curve_PSO = zeros(N_rep, max_iter);
all_curve_GA  = zeros(N_rep, max_iter);

%% ==== 2. 多次实验主循环 ====
for rep = 1:N_rep
    fprintf('\n第%d/%d次实验：\n', rep, N_rep);

    %% ==== DBO ====
    disp('DBO优化中...');
    tic;
    [~, fit_dbo, curve_dbo] = DBO(fitness_func, lb, ub, dim, pop_size, max_iter);
    all_time_DBO(rep) = toc;
    all_fit_DBO(rep) = fit_dbo;
    all_curve_DBO(rep,:) = curve_dbo;

    %% ==== PSO ====
    disp('PSO优化中...');
    tic;
    [~, fit_pso, curve_pso] = PSO(fitness_func, lb, ub, dim, pop_size, max_iter);
    all_time_PSO(rep) = toc;
    all_fit_PSO(rep) = fit_pso;
    all_curve_PSO(rep,:) = curve_pso;

    %% ==== GA ====
    disp('GA优化中...');
    tic;
    [~, fit_ga, curve_ga] = GA(fitness_func, lb, ub, dim, pop_size, max_iter);
    all_time_GA(rep) = toc;
    all_fit_GA(rep) = fit_ga;
    all_curve_GA(rep,:) = curve_ga;
end

%% ==== 3. 结果统计 ====
% 平均最优适应度、平均时间
mean_fit_DBO = mean(all_fit_DBO); mean_time_DBO = mean(all_time_DBO);
mean_fit_PSO = mean(all_fit_PSO); mean_time_PSO = mean(all_time_PSO);
mean_fit_GA  = mean(all_fit_GA);  mean_time_GA  = mean(all_time_GA);

% 最后一次实验的收敛曲线
curve_DBO = all_curve_DBO(end,:);
curve_PSO = all_curve_PSO(end,:);
curve_GA  = all_curve_GA(end,:);

fprintf('\n------------- 多次平均优化算法性能对比 -------------\n');
fprintf('算法\t\t平均最优适应度\t平均运行时间(s)\n');
fprintf('DBO\t\t%.4f\t\t%.2f\n', mean_fit_DBO, mean_time_DBO);
fprintf('PSO\t\t%.4f\t\t%.2f\n', mean_fit_PSO, mean_time_PSO);
fprintf('GA\t\t%.4f\t\t%.2f\n',  mean_fit_GA,  mean_time_GA);

%% ==== 4. 平均收敛曲线可视化 ====
figure;
plot(mean(all_curve_DBO,1),'-r','LineWidth',2); hold on;
plot(mean(all_curve_PSO,1),'-b','LineWidth',2);
plot(mean(all_curve_GA,1), '-g','LineWidth',2);
xlabel('迭代次数'); ylabel('平均最优适应度');
legend('DBO','PSO','GA');
title(['三种优化算法平均收敛曲线（重复' num2str(N_rep) '次）']);
grid on;