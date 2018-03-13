
clear
close all

Ncv = 3;
qcv = 1/Ncv;

N = 3;
q = 1/N;
addpath('glmnet')

load('data_fabian/neurons08.mat');
n_ori = length(ori);
time_idx = (times >= 50 & times <= 250);
%time_idx = (times >= 0 | times <= 0);
mi =  NaN(nNeurons,N);
mi_all = NaN(1,N);
ori1 = 1;
ori2 = 3;

x_all = [];

warning off

jc = 2;
for b = 2:nNeurons
    x =  squeeze(permute(cat(3,feat{jc,ori1}(b,time_idx,:), ...
        feat{jc,ori2}(b,time_idx,:)), [3 1 2]));
    x_all = [x_all, x];
    
    y = [ones(nTrials,1)+1; ones(nTrials,1)];
    fprintf(' %i', b);
    for j = 1:N
        fprintf('.');
        % prepare training and test set
        [train_idx, test_idx] = cv_split(q,y);
        x_train = x(train_idx,:); y_train = y(train_idx);
        x_test = x(test_idx,:); y_test = y(test_idx);
        classifier(j) = model_selection_cv(x_train,y_train,Ncv,qcv,0);
        classifier(j) = test(classifier(j),x_test, y_test);
        mi(b,j) = compute_MI(classifier(j).yest_test, y_test);
    end
end

fprintf(' all\n');
for j = 1:N
    fprintf('.');
    % prepare training and test set
    [train_idx, test_idx] = cv_split(q,y);
    x_train = x_all(train_idx,:); y_train = y(train_idx);
    x_test = x_all(test_idx,:); y_test = y(test_idx);
    classifier(j) = model_selection_cv(x_train,y_train,Ncv,qcv,0);
    classifier(j) = test(classifier(j),x_test, y_test);
    mi_all(j) = compute_MI(classifier(j).yest_test, y_test);
end

warning on

figure()
m_si = nanmean(mi,2);
ms_si = nanmean(cumsum(mi(2:end,:),1),2);

s_si = nanstd(m_si);
m_si =nanmean(m_si);

m_pop = nanmean(mi_all);

bar([1,2],[m_si,m_pop],'Facecolor',[.7,.7,.7]);
hold on
for i = length(ms_si):-1:1
    bar(3,ms_si(i),'Facecolor',[.5,1. - mod(i,3)/3,mod(i+2,3)/3]);
end
errorbar(1,m_si,s_si,'.k','LineWidth',2);
axis([0,4,0,3])
set(gca,'Xtick',[1,2,3])
set(gca,'Xticklabel',{'single','population','sum'})
ylabel('mutual information [bits]');
title('single units vs. population');


