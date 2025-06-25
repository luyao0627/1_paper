function [gbest, gbestval, curve] = GA(fitness_func, lb, ub, dim, pop_size, max_iter)
% 遗传算法（GA, Genetic Algorithm）简化实现
% 输入参数说明：
%   fitness_func : 适应度函数句柄
%   lb           : 参数下界 (1 x dim)
%   ub           : 参数上界 (1 x dim)
%   dim          : 优化参数个数
%   pop_size     : 种群规模
%   max_iter     : 最大进化代数
% 输出：
%   gbest    : 全局最优参数
%   gbestval : 全局最优适应度
%   curve    : 收敛曲线

pc = 0.9;     % 交叉概率
pm = 0.1;     % 变异概率

% 初始化种群，每行为一个个体
X = rand(pop_size,dim) .* (ub-lb) + lb;

% 计算初始种群适应度
fit = arrayfun(@(i)fitness_func(X(i,:)), 1:pop_size)';
[gbestval, idx] = max(fit); % 全局最优适应度
gbest = X(idx,:);           % 全局最优参数
curve = zeros(max_iter,1);  % 收敛曲线

for t = 1:max_iter
    % 选择操作：锦标赛选择
    idx_sel = randi(pop_size, pop_size, 2); % 每行两个候选
    sel = fit(idx_sel(:,1)) > fit(idx_sel(:,2));
    parents = X(idx_sel(sub2ind([pop_size,2], (1:pop_size)', sel+1)), :);

    % 交叉操作：单点交叉
    offspring = parents;
    for i = 1:2:pop_size-1
        if rand < pc
            point = randi(dim-1); % 随机选择交叉点
            temp = offspring(i, point+1:end);
            offspring(i, point+1:end) = offspring(i+1, point+1:end);
            offspring(i+1, point+1:end) = temp;
        end
    end

    % 变异操作：每个基因有概率变异
    for i = 1:pop_size
        for j = 1:dim
            if rand < pm
                offspring(i,j) = lb(j) + rand * (ub(j) - lb(j));
            end
        end
    end

    % 计算子代适应度
    fit_off = arrayfun(@(i)fitness_func(offspring(i,:)), 1:pop_size)';

    % 环境选择：父代+子代合并，选最优pop_size个
    X = [X; offspring];
    fit_all = [fit; fit_off];
    [~, idx_sort] = sort(fit_all, 'descend');
    X = X(idx_sort(1:pop_size), :);
    fit = fit_all(idx_sort(1:pop_size));

    % 更新全局最优
    [bestval, idx] = max(fit);
    if bestval > gbestval
        gbestval = bestval;
        gbest = X(idx,:);
    end
    curve(t) = gbestval; % 记录本代最优适应度
end
end