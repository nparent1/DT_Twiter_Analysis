%% Load Data
% opts = detectImportOptions('dataset.xlsx');
% opts.SelectedVariableNames = [8 12:334];
% X = readmatrix('dataset.xlsx',opts);
% opts.SelectedVariableNames = 2;
% Y = readmatrix('dataset.xlsx',opts);
load('X.mat');
load('Y.mat');
s = string(Y);
id = s == "Twitter for iPhone" | s == "Twitter for Android";
s = s(id);
X = X(id,:);
Y = s == "Twitter for iPhone";

%% Mean normalize X
non_present = max(X) == 0;
X = X(:,~non_present);
X = (X - repmat(mean(X), size(X,1), 1)) ./ repmat(2*std(X), size(X,1), 1);
X = [ones(size(X,1),1) X];
%% Divide data
m = size(X,1);
[trainInd,valInd,testInd] = dividerand(m,.6,.2,.2);
X_train = X(trainInd,:);
Y_train = Y(trainInd,:);
X_val = X(valInd,:);
Y_val = Y(valInd,:);
X_test = X(testInd,:);
Y_test = Y(testInd,:);


%% Logistic Regression - Train theta
initial_theta = zeros(size(X_train,2),1);
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400); 
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X_train, Y_train,1)), initial_theta, options);

%% Predict on CV set
p = predict(theta, X_val);
success = p == Y_val;
success = success(success);
success_rate = length(success)/length(p);

