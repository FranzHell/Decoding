function mi = compute_MI(Ypred, Y)

y = sort(unique(Y(:)'));
n = length(y);

P = NaN(n,n);
p = NaN*y';
q = NaN*y';
for i = 1:n
    for j = 1:n
        P(i,j) = nanmean(Ypred == y(i) &  Y == y(j));
    end
    p(i) = mean(Y==y(i));
    q(i) = nanmean(Ypred==y(i));
end

mi = -nansum(p.*log2(p)) -nansum(q.*log2(q)) + nansum(P(:).*log2(P(:)));

