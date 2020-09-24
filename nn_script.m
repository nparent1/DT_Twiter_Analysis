% nn script
hidden_layer_size = 40;
input_layer_size = 308;
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
[trainInd,valInd,testInd] = dividerand(m,.8,0,.2);
X_train = X(trainInd,:);
Y_train = Y(trainInd,:);
X_val = X(valInd,:);
Y_val = Y(valInd,:);
X_test = X(testInd,:);
Y_test = Y(testInd,:);

%% Initialize Thetas
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, 1);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Train
options = optimset('MaxIter', 60);
lambda = 3;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, X_train, Y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), 1, (hidden_layer_size + 1));
 
pred_train = predict_nn(Theta1, Theta2, X_train);
train_accuracy = mean(double(pred_train == Y_train)) * 100;

pred_val = predict_nn(Theta1, Theta2, X_val);
val_accuracy = mean(double(pred_val == Y_val)) * 100;

pred_test = predict_nn(Theta1, Theta2, X_test);
test_accuracy = mean(double(pred_test == Y_test)) * 100;