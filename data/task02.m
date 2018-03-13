% perform a leave one out model selection
clear
close all
addpath('glmnet');

load task01.mat

classifier = model_selection_cv(X, Y);



%% plot
fit = classifier.model;
Ypred = glmnetPredict(fit,'class',Xtest);

[x,y] = meshgrid(linspace(-5,5,50));
lb = glmnetPredict(fit, 'response', [x(:),y(:)]);
figure
size(lb)
surface(x,y,reshape(lb,size(x))-1,'FaceAlpha',1);
hold on
contour(x,y,reshape(lb,size(x)),[.5,.5],'LineWidth',3,'Color','b');
shading interp
colorbar
colormap jet
plot(X(Y==1,1),X(Y==1,2),'+g','MarkerSize',5,'LineWidth',3);
plot(X(Y==2,1),X(Y==2,2),'xr','MarkerSize',5,'LineWidth',3);
plot(Xtest(Ypred==1,1),Xtest(Ypred==1,2),'og','MarkerSize',5,'LineWidth',3);
plot(Xtest(Ypred==2,1),Xtest(Ypred==2,2),'or','MarkerSize',5,'LineWidth',3);
xlabel('x_1')
ylabel('x_2')
hold off