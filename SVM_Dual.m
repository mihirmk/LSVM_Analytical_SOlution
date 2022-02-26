% test
close all
clear all
clc

% Loading the data
load linear_svm.mat

% The training data
X_train = X_train;
y_train = labels_train;
% The testing data
X_test = X_test;
y_test = labels_test;


%% Dual Problem SOlution
tic
C  = 150;
gamma = 0.05;
[n,d]=size(X_train);

% Formulating Standard QP form
Q = diag(y_train)*X_train*(X_train')*diag(y_train);
p = -1*ones(n,1);
b = [(C/(gamma*n)).*ones(n,1); zeros(n,1)];
A = [eye(n);-1*eye(n) ];

% Barrier Method Initializations
s0_dual= (C/(1.5*gamma*n)).* ones(n,1);
mu  = 10;
tol = 0.0000001;
[s_dual,s_hist,tol_gap,obj] = newton_barrier(Q,p,A,b,s0_dual,mu,tol);

% Hyperplane Results of the Dual Solution
w_d = X_train'*diag(y_train)*s_dual; 

b     = mean(y_train - X_train*w_d); 

toc

%% Plot Functions

x1 = linspace(0,4,100);
x2 = -(b + w_d(1,1) * x1 ) /w_d(2,1) ;

% Response on Training Data
figure
class1 = (labels_train > 0);
class2 = (labels_train < 0);
scatter(X_train(class1,1),X_train(class1,2),'DisplayName', 'Class 1')
hold on
scatter(X_train(class2,1),X_train(class2,2),'DisplayName', 'Class 2')
xlabel('x_1'); ylabel('x_2');
xlim([0,4]); ylim([0,4]);
plot(x1 ,x2 ,  'r' ,'linewidth',1,'DisplayName', 'Hyperplane');
grid minor
legend('Location','best')
saveas(gcf,'d_train.png')

% Response on Training Data
figure
class1 = (labels_test > 0);
class2 = (labels_test < 0);
scatter(X_test(class1,1),X_test(class1,2),'DisplayName', 'Class 1')
hold on
scatter(X_test(class2,1),X_test(class2,2),'DisplayName', 'Class 2')
xlabel('x_1'); ylabel('x_2');
xlim([0,4]); ylim([0,4]);
plot(x1 ,x2 ,  'r' ,'linewidth',1,'DisplayName', 'Hyperplane');
grid minor
legend('Location','best')
saveas(gcf,'d_test.png')

% Plot of Primal Problem Slack
figure
semilogy(mean(s_hist),'linewidth',1);
grid minor
xlabel('Iteration'); ylabel('z');
title('Dual Problem Slack')
saveas(gcf,'d_w.png')

% Plot of Objective Value
figure
semilogy(obj ,'linewidth',1);
grid minor
xlabel('Iteration'); ylabel('Objective Value');
title('Trajectory of Objective Value')
saveas(gcf,'d_obj.png')

% Plot of Gaps at Each step
figure
semilogy(tol_gap ,'linewidth',1);
grid minor
xlabel('Iteration'); ylabel('Tolerace Gap m/t');
title('Trajectory of Tolerance Gap')
saveas(gcf,'d_gap.png')