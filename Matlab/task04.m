% compute lower bound on MI with a single neuron
clear
close all

Ncv = 10;
qcv = 1/Ncv;

N = 5;
q = 1/2;
addpath('glmnet')

load('neurons08.mat');
n_ori = length(ori);

mi =  NaN(length(contrast),nBins,N);
ori1 = 1;
ori2 = 5;

for b = 1:nBins
    fprintf('Bin %02i\n', b);
    for jc = 1:length(contrast)
        Y = [];
        x =  permute(cat(3,feat{jc,ori1}(:,b,:), ...
            feat{jc,ori2}(:,b,:)), [3 1 2]);
        y = [ones(nTrials,1)+1; ones(nTrials,1)];
        for j = 1:N
            % prepare training and test set
            [train_idx, test_idx] = cv_split(q,y);
            x_train = x(train_idx,:); y_train = y(train_idx);
            x_test = x(test_idx,:); y_test = y(test_idx);
            Y = [Y;y_test];
            
            classifier(j) = model_selection_cv(x_train,y_train,Ncv,qcv,0);
            classifier(j) = test(classifier(j),x_test, y_test);
            mi(jc,b,j) = compute_MI(classifier(j).yest_test, y_test);
        end
        
    end
end

figure()
mmi = mean(mi,3);
smi = std(mi,[],3);

fill([times, fliplr(times)],[mmi(1,:)+smi(1,:), fliplr(mmi(1,:)-smi(1,:))],...
    'k','FaceAlpha',.2);
hold on
fill([times, fliplr(times)],[mmi(2,:)+smi(2,:), fliplr(mmi(2,:)-smi(2,:))],...
    'b','FaceAlpha',.5);
plot(times, mmi(1,:),'k-','LineWidth',2);
plot(times,mmi(2,:),'b-','LineWidth',2);
hold off
legend(sprintf('std contrast %.2f', contrast(1)), ...
    sprintf('std contrast %.2f', contrast(2)),...
    sprintf('mean contrast %.2f',contrast(1)), ...
    sprintf('mean contrast %.2f',contrast(2)));
xlabel('time [ms]');
ylabel('mutual information [bits]');
title(sprintf('lower bound on MI for \theta=%.2f vs \theta=%.2f',ori(ori1), ori(ori2)));

