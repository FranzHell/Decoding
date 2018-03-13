% train and test a single classifier
addpath('glmnet');
load task01.mat
opt.alpha = 0;     % trade off between L1 and L2 regularization
opt.lambda = 2.3;   % regularization parameter
opt = glmnetSet(opt);

fit = glmnet(X,Y,'binomial', opt);



Yhat = glmnetPredict(fit, 'class', Xtest);

