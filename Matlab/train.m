function classifier = train(X,Y,alpha, lambda)

if nargin < 4
    s = mean(diag(cov(X)));
    lambda = s*2.^[-8:2];
end

classifier = struct;
for i = 1:length(alpha)
    for j = 1:length(lambda)
        opt.alpha = alpha(i);     % trade off between L1 and L2 regularization
        opt.lambda = lambda(j);  % range of regularization path
        opt = glmnetSet(opt);
        
        try
            classifier(i,j).trainingFailed = true;
            fit = glmnet(X,Y,'binomial',opt);
            
            classifier(i,j).model = fit;
            classifier(i,j).yest_train = glmnetPredict(fit,'class',X);
            classifier(i,j).p_train = glmnetPredict(fit,'response', X);
            classifier(i,j).mi_train = compute_MI(classifier(i,j).yest_train, Y);
            classifier(i,j).error_train = mean(classifier(i,j).yest_train ~= Y);
            classifier(i,j).accuracy_train = 1-classifier(i,j).error_train;
                
            classifier(i,j).param = opt;
            classifier(i,j).trainingFailed = false;
        catch
            classifier(i,j).model = NaN;
            classifier(i,j).yest_train = NaN*Y;
            classifier(i,j).p_train = NaN*Y;
            classifier(i,j).error_train = NaN;
            classifier(i,j).accuracy_train =NaN;
            classifier(i,j).mi_train = NaN;
            
            classifier(i,j).param = opt;
            classifier(i,j).trainingFailed = true;
        end
    end
end