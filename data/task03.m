


acc = zeros(25,1);
mi = zeros(25,1);

for i = 1:25
    
    load(sprintf('task03_%02i.mat',i));
    classifier = model_selection_cv(X,Y,10, .2, 0);
    classifier = test(classifier, Xval, Yval);
    
    acc(i) = classifier.accuracy_test;
    mi(i) = classifier.mi_test;
end

figure
plot(acc)
figure


plot(bayes_err, acc,'-k','LineWidth',2);
hold on
plot(bayes_err, mi,'-b','LineWidth',2)
hold off
legend('accuracy','mutual information [bits]')
xlabel('Bayes error');
ylabel('Mutual information/classification error')