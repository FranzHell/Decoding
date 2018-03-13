function classifiers = train_cv(X, Y, N, q, alpha, lambda)
if nargin < 5
    alpha = linspace(0,1,10);
end

if nargin < 6
    s = mean(diag(cov(X)));
    lambda = s*2.^[-8:2];
end


classifiers = {};
for i = 1:N
    %fprintf('.');
    [train_idx, test_idx] = cv_split(q,Y);
    classifiers{i} = train(X(train_idx,:),Y(train_idx), alpha, lambda);
    classifiers{i} = test(classifiers{i}, X(test_idx,:),Y(test_idx));
end
classifiers = cat(3, classifiers{:});

