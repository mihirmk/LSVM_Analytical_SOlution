clear all
clc
close all

%% SVM Problem
load('linear_svm.mat')
n = length(X_train);
 
cvx_begin
variables w1(2,1) b1(1)
    minimize (0.5*w1'*w1)
    subject to 
    ones(1,100) - labels_train'.*(w1'*X_train' + b1*ones(1,100)) <= 0;
cvx_end

%% Plotting Function


% Plot of Classifier Training Data
class1 = (labels_train > 0);
class2 = (labels_train < 0);
figure
scatter(X_train(class1,1),X_train(class1,2),'filled','DisplayName','Class 1')
hold on
scatter(X_train(class2,1),X_train(class2,2),'filled' ,'DisplayName','Class 2')
x1 = linspace(0,4,100);
xlim([0,4]); ylim([0,4]);

w = w1; b = b1;
x2 = -(b + w(1,1) * x1 ) /w(2,1);
plot(x1 ,x2,'DisplayName','Hyperplane','Linewidth',1);
title('Training Data')
grid on
legend
saveas(gcf,'cvx_train.png')

% Plot of Classifier Test Data
class1 = (labels_test > 0);
class2 = (labels_test < 0);
figure
scatter(X_test(class1,1),X_test(class1,2),'filled','DisplayName','Class 1')
hold on
scatter(X_test(class2,1),X_test(class2,2),'filled','DisplayName','Class 2')

xlim([0,4]); ylim([0,4]);

w = w1; b = b1;
x2 = -(b + w(1,1) * x1 ) /w(2,1);
plot(x1 ,x2,'DisplayName','Hyperplane','Linewidth',1);
title('Test Data')
grid on
legend
saveas(gcf,'cvx_test.png')