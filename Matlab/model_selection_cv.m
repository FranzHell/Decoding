function classifier = model_selection_cv(X, Y, N, q, alpha, lambda, objective)

m = size(X,1);

if nargin < 3
    N = 10;
end

if nargin < 4
    q = 0.2;
end

if nargin < 5
    alpha = linspace(0,1,10);
end

if nargin < 6
    s = mean(diag(cov(X)));
    lambda = s*2.^[-8:0];
end

if nargin < 7
    objective = 'accuracy_test';
end

%fprintf('Cross validation ')

classifiers = train_cv(X,Y,N,q,alpha,lambda);
O = reshape(cat(1,classifiers.(objective)), size(classifiers));

%fprintf('\n');
mO = mean(O,3);

if all(isnan(mO))
    classifier = classifiers(1,1,1);
else
    
    [i_idx, j_idx] = find(mO == max(mO(:)));
    % select one model from all the ones that achieve the maximum. Since larger
    % column indices correspond to stronger regularized models, we choose the
    % most strongly regularized one.
    [dummy, i] = max(j_idx);
    
    classifier = classifiers(i_idx(i), j_idx(i),1);
end

classifier = train(X,Y, classifier.param.alpha, classifier.param.lambda);
classifier = test(classifier, X,Y);

