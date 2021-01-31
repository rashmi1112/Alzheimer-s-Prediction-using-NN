%% Initialization
clear ; close all; clc

%% Setup the parameters 
input_layer_size  = 15 % 15 Input features for each row
hidden_layer_size = 14   % 15 hidden nodes
hidden_layer1_size = 10
num_labels =  4        % One each for every CDR category (0,0.5,1,2)

%% Load Training Data 
fprintf('\nLoading data from the files ...\n')
Long_data = csvread('oasis_longitudinal.csv');
CS_data = csvread('oasis_cross-sectional.csv');
MF_data = fileread("Gender.txt");
MF_data = strsplit (MF_data, "\n");

MF_vec = labelEncoding(MF_data);

%% Data pre-processing
fprintf('\nPre-Processing data ...\n')
M = preprocessData(Long_data,CS_data,MF_vec);


%% Splitting the data into train, cv and train. 
fprintf('\nSplitting data into train,cross-validation and test sets ...\n')
per_train = 0.6;
per_CV = 0.2;
[m,n] = size(M);
random_idx = randperm(m);
train_data = random_idx(1:round(per_train*m));
CV_data = random_idx(round(per_train*m)+1:round(per_train*m)+round(per_CV*m));
test_data = random_idx(round(per_train*m)+round(per_CV*m)+1:end);
data_train = M(train_data,:);
data_cv = M(CV_data,:);
data_test = M(test_data,:);

size_train = size(data_train);
size_cv = size(data_cv);
size_test = size(data_test);

X = data_train(:,1:end-1);
y = data_train(:,end);


Xval = data_cv(:,1:end-1);
yval = data_cv(:,end);

Xtest = data_test(:,1:end-1);
ytest = data_test(:,end);

%% Normalize the features since we have a variety of ranges throughout.
[X_norm, mu, sigma] = featureNormalize(X);
size_xnormout = size(X_norm);
[Xval_norm, mu1, sigma1] = featureNormalize(Xval);
[Xtest_norm, mu2, sigma2] = featureNormalize(Xtest); 


%% Initialization of the weights for the neural network
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeTheta(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeTheta(hidden_layer_size, hidden_layer1_size);
initial_Theta3 = randInitializeTheta(hidden_layer1_size, num_labels);

%% Unroll Parameters 
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];

%% Training the neural network 
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 1000);
lambda = 0.02

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer1_size, ...
                                   num_labels, ...
                                   X_norm, y, lambda);
fprintf('\nTraining Neural Network... \n')
             
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
%% Obtain Theta1 and Theta2 back from nn_params

shape1 = hidden_layer_size * (input_layer_size + 1);
shape2 = hidden_layer1_size*(hidden_layer_size+1);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
            
Theta2 = reshape(nn_params(1+shape1:shape1+shape2),...
                 hidden_layer1_size,(hidden_layer_size+1));
                 
Theta3 = reshape(nn_params(1+shape1+shape2:end),...
                 num_labels,(hidden_layer1_size+1));


%% Prediction of the CDR using the obtained optimized parameters

pred = NNpredict(Theta1, Theta2, Theta3, X_norm);
fprintf('\nTraining Set Accuracy: %f%%\n', mean(double(pred == y)) * 100);

%% Prediction of Cross Validation set.

pred_cv = NNpredict(Theta1, Theta2, Theta3, Xval_norm);
fprintf('\nCross Validation Set Accuracy: %f%%\n', mean(double(pred_cv == yval)) * 100);

%% Prediction of Test set.
pred_test = NNpredict(Theta1, Theta2, Theta3, Xtest_norm);
fprintf('\nTest Set Accuracy: %f%%\n', mean(double(pred_test == ytest)) * 100);


