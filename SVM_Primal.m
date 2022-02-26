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


%% Primal Problem Solution
tic
C  = 40;
gamma = 0.001;
[n,d]=size(X_train);

% Formulating Standard QP form
X = [ones(n,1) X_train];
Q = [eye(d+1) zeros(d+1,n); zeros(n,d+1) zeros(n,n)];
p = [zeros(d+1,1);(C/(gamma*n))*ones(n,1)];
b = [-1*ones(n,1); zeros(n,1)];
A = [-diag(y_train)*X -eye(n);zeros(n,d+1) -eye(n)];

% Barrier Method Initializations
w0_primal= [zeros(d+1,1); (C/(1.5*gamma*n)).* ones(n,1)];
mu  = 10;
tol = 0.0000001;
[w_primal,w_hist,tol_gap,obj] = newton_barrier(Q,p,A,b,w0_primal,mu,tol);

% Hyperplane Results of the Dual Solution
w_p = w_primal(1:3);

w_h = w_hist(1:3,:);

toc

%% Plot Functions

x1 = linspace(0,4,100);
x2 = -(w_p(1,1) + w_p(2,1) * x1 ) /w_p(3,1) ;

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
title('Training Data')
legend('Location','best')
saveas(gcf,'p_train.png')

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
title('Test Data')
legend('Location','best')
saveas(gcf,'p_test.png')

% Plot of Primal Problem Slack
figure
semilogy(mean(w_hist(4:end,:)),'linewidth',1);
grid minor
xlabel('Iteration'); ylabel('z');
title('Primary Problem Slack')
saveas(gcf,'p_w.png')

% Plot of Objective Value
figure
semilogy(obj ,'linewidth',1);
grid minor
xlabel('Iteration'); ylabel('Objective Value');
title('Trajectory of Objective Value')
saveas(gcf,'p_obj.png')

% Plot of Gaps at Each step
figure
semilogy(tol_gap ,'linewidth',1);
grid minor
xlabel('Iteration'); ylabel('Tolerace Gap m/t');
title('Trajectory of Tolerance Gap')
saveas(gcf,'p_gap.png')