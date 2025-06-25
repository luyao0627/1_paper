function [gbest, gbestval, curve] = PSO(fitness_func, lb, ub, dim, pop_size, max_iter)
% 粒子群优化算法（PSO, Particle Swarm Optimization）简化实现
% 输入参数说明：
%   fitness_func : 适应度函数句柄
%   lb           : 参数下界 (1 x dim)
%   ub           : 参数上界 (1 x dim)
%   dim          : 优化参数个数
%   pop_size     : 粒子数量（种群规模）
%   max_iter     : 最大迭代次数
% 输出：
%   gbest    : 全局最优参数
%   gbestval : 全局最优适应度
%   curve    : 收敛曲线

w = 0.7;      % 惯性权重
c1 = 1.5;     % 个体学习因子
c2 = 1.5;     % 社会学习因子

% 初始化粒子位置和速度
X = rand(pop_size,dim) .* (ub-lb) + lb; % 每行是一个粒子
V = zeros(pop_size,dim);                % 粒子速度

pbest = X; % 个体历史最优位置
% 计算初始适应度
pbestval = arrayfun(@(i)fitness_func(X(i,:)), 1:pop_size)'; 
[gbestval, idx] = max(pbestval);   % 全局最优适应度和粒子
gbest = X(idx,:);
curve = zeros(max_iter,1);         % 收敛曲线

for t = 1:max_iter
    for i = 1:pop_size
        % 速度更新
        V(i,:) = w*V(i,:) + c1*rand(1,dim).*(pbest(i,:)-X(i,:)) + c2*rand(1,dim).*(gbest-X(i,:));
        % 位置更新
        X(i,:) = X(i,:) + V(i,:);
        % 边界处理
        X(i,:) = max(lb, min(ub, X(i,:)));
        % 计算适应度
        fit = fitness_func(X(i,:));
        % 个体最优更新
        if fit > pbestval(i)
            pbest(i,:) = X(i,:);
            pbestval(i) = fit;
        end
    end
    % 全局最优更新
    [bestval, idx] = max(pbestval);
    if bestval > gbestval
        gbestval = bestval;
        gbest = pbest(idx,:);
    end
    curve(t) = gbestval; % 记录本代最优
end
end