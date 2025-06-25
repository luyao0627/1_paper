function pe = permutation_entropy(sig, m, tau)
% 排列熵
N = length(sig);
patterns = zeros(N-(m-1)*tau, m);
for i=1:N-(m-1)*tau
    patterns(i,:) = sig(i:tau:i+(m-1)*tau);
end
[~, perm] = sort(patterns,2);
[uniq, ~, idx] = unique(perm, 'rows');
p = histc(idx, 1:size(uniq,1))/size(patterns,1);
pe = -sum(p.*log2(p+eps));
end