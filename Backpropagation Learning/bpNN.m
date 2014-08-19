%% Kaggle Digit Recognizer

%  Instructions
%  ------------
% 
%  

%% ======================================================================
%  STEP 0: Load the data and setup the parameters
%  Initialization
clear ; close all; clc

addpath(genpath('../lib'));
debug = 1;

trainData   = loadMNISTImages('../data/mnist/train-images.idx3-ubyte'); % inputSize x numCases
trainData = trainData';
trainLabels = loadMNISTLabels('../data/mnist/train-labels.idx1-ubyte');
trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1


[numCases,inputSize]  = size(trainData);  
hiddenSize = 200;            % 25 hidden units
num_labels = 10;           % 10 labels, from 1 to 10   
                                      % (note that we have mapped "0" to label 10)
 
testData = loadMNISTImages('../data/mnist/t10k-images.idx3-ubyte');  % inputSize x numCases
testData = testData';
testLabels = loadMNISTLabels('../data/mnist/t10k-labels.idx1-ubyte');
testLabels(testLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

    
% STEP 1 : Build and Train the Neural Network

%  Randomly initialize the weights
%  the hidden layer weights size hiddenSize x  (inputSize + 1)
%  the output layer weights size num_labels x (hiddenSize +1)
%  +1 is for bias
initial_Theta1 = randInitWeights(hiddenSize, inputSize);
initial_Theta2 = randInitWeights(num_labels, hiddenSize);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%  lambda for weight decay
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   inputSize, ...
                                   hiddenSize, ...
                                   num_labels, trainData, trainLabels, lambda);

%  set MaxIter to 100, it can be changed to larger number. e.g. 400
options = optimset('MaxIter', 400);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hiddenSize * (inputSize + 1)), ...
                 hiddenSize, (inputSize + 1));

Theta2 = reshape(nn_params((1 + (hiddenSize * (inputSize + 1))):end), ...
                 num_labels, (hiddenSize + 1));

 %% Step 2 : Predict
 
pred = predict(Theta1, Theta2, testData);

if debug == 1
    fprintf('\nTotal number: %f Training Set Accuracy: %f\n', length(testLabels), mean(double(pred == testLabels)) * 100);
    
    % fprintf('\nErrors \n');
    pred_error = [pred(any((pred ~= testLabels),2)),testLabels(any((pred ~= testLabels),2))];

    hist(testLabels(any((pred ~= testLabels),2)))
end                                 