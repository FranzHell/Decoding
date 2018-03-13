function [train_idx, test_idx] = cv_split(q,Y)
y = unique(Y);
ret = struct;
for i = 1:length(y)
    ret(i).idx = find(Y == y(i));
    ret(i).n = length(ret(i).idx);
    tmp = randperm(ret(i).n);
    ret(i).test_idx = ret(i).idx(tmp(1:ceil(ret(i).n*q)));
    ret(i).train_idx = ret(i).idx(tmp(ceil(ret(i).n*q)+1:end));
end
test_idx = cat(1,ret.test_idx);
train_idx = cat(1,ret.train_idx);
