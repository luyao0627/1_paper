function [gbest, gbestval, curve] = DBO(fitness_func, lb, ub, dim, pop_size, max_iter)
% Dung Beetle Optimizer (DBO) 简化实现
% 输入参数说明：
%   fitness_func : 适应度函数句柄
%   lb           : 参数下界 (1 x dim)
%   ub           : 参数上界 (1 x dim)
%   dim          : 优化参数个数
%   pop_size     : 粒子/个体数量
%   max_iter     : 最大迭代次数
% 输出：
%   gbest    : 全局最优参数
%   gbestval : 全局最优适应度
%   curve    : 收敛曲线

% 初始化种群
X = rand(pop_size,dim) .* (ub-lb) + lb;
fit = zeros(pop_size,1);
for i = 1:pop_size
    fit(i) = fitness_func(X(i,:));
end
[gbestval, idx] = max(fit);
gbest = X(idx,:);
curve = zeros(max_iter,1);

for t = 1:max_iter
    for i = 1:pop_size
        for j = 1:dim
            if rand < 0.5
                % 利用全局最优引导更新
                X(i,j) = X(i,j) + randn() * (gbest(j) - X(i,j));
            else
                % 利用种群内其它个体引导更新
                r = randi(pop_size);
                X(i,j) = X(i,j) + rand() * (X(r,j) - X(i,j));
            end
            % 边界处理
            X(i,j) = max(lb(j), min(ub(j), X(i,j)));
        end
        % 更新个体适应度
        newfit = fitness_func(X(i,:));
        if newfit > fit(i)
            fit(i) = newfit;
        end
    end
    % 更新全局最优
    [bestval, idx] = max(fit);
    if bestval > gbestval
        gbestval = bestval;
        gbest = X(idx,:);
    end
    curve(t) = gbestval;
end
end