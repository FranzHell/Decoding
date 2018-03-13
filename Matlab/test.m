function classifier = test(classifier, X, Y)

[m,n] = size(classifier);
for i = 1:m
    for j = 1:n
        if ~classifier(i,j).trainingFailed
            classifier(i,j).yest_test = glmnetPredict(classifier(i,j).model,'class',X);
            classifier(i,j).error_test = mean(classifier(i,j).yest_test ~= Y);
            classifier(i,j).accuracy_test = 1-classifier(i,j).error_test;
            classifier(i,j).p_test = glmnetPredict(classifier(i,j).model,'response',X);
            classifier(i,j).mi_test = compute_MI(classifier(i,j).yest_test,Y);
        else
            classifier(i,j).yest_test = Y*NaN;
            classifier(i,j).error_test = NaN;
            classifier(i,j).accuracy_test = NaN;
            classifier(i,j).p_test = Y*NaN;
            classifier(i,j).mi_test = NaN;
        end
    end
end