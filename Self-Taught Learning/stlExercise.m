%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%

% add lib path
addpath(genpath('../lib'));

%% eyulush : add some control for self-training with 10 digits

numSelfTrainingDigits = 10;
unlabeledDataSize = 10000;

%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize  = 28 * 28;
numLabels  = numSelfTrainingDigits;
hiddenSize = 200;
sparsityParam = 0.1; % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             %  in the lecture notes). 
lambda = 3e-3;       % weight decay parameter       
beta = 3;            % weight of sparsity penalty term   
maxIter = 400;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

if numSelfTrainingDigits == 5
    % Load MNIST database files
    mnistData   = loadMNISTImages('../data/mnist/train-images.idx3-ubyte');
    mnistLabels = loadMNISTLabels('../data/mnist/train-labels.idx1-ubyte');
    
    % Set Unlabeled Set (All Images)
    % Simulate a Labeled and Unlabeled set
    labeledSet   = find(mnistLabels >= 0 & mnistLabels <= 4);
    unlabeledSet = find(mnistLabels >= 5);

    numTrain = round(numel(labeledSet)/2);
    trainSet = labeledSet(1:numTrain);
    testSet  = labeledSet(numTrain+1:end);

    unlabeledData = mnistData(:, unlabeledSet);

    trainData   = mnistData(:, trainSet);
    trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-5

    testData   = mnistData(:, testSet);
    testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-5

    % Output Some Statistics
    fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
    fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
    fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));
end

if numSelfTrainingDigits == 10
    % Load MNIST database files
    trainData   = loadMNISTImages('../data/mnist/train-images.idx3-ubyte');
    trainLabels = loadMNISTLabels('../data/mnist/train-labels.idx1-ubyte');
    trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
    
    testData = loadMNISTImages('../data/mnist/t10k-images.idx3-ubyte');
    testLabels = loadMNISTLabels('../data/mnist/t10k-labels.idx1-ubyte');
    testLabels(testLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

    unlabeledData = trainData(:, 1:unlabeledDataSize);  
    
    % Output Some Statistics
    fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
    fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
    fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));
end

%% ======================================================================
%  STEP 2: Train the sparse autoencoder
%  This trains the sparse autoencoder on the unlabeled training
%  images. 

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, inputSize);

%% ----------------- YOUR CODE HERE ----------------------
%  Find opttheta by running the sparse autoencoder on
%  unlabeledTrainingImages

opttheta = theta; 

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

% use unlabeledData for self-unsupervised-training
[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, unlabeledData), ...
                                   opttheta, options);

%% -----------------------------------------------------
                          
% Visualize weights
W1 = reshape(opttheta(1:hiddenSize * inputSize), hiddenSize, inputSize);
display_network(W1');

%%======================================================================
%% STEP 3: Extract Features from the Supervised Dataset
%  
%  You need to complete the code in feedForwardAutoencoder.m so that the 
%  following command will extract features from the data.

trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, ...
                                       testData);

%%======================================================================
%% STEP 4: Train the softmax classifier

softmaxModel = struct;  
%% ----------------- YOUR CODE HERE ----------------------
%  Use softmaxTrain.m from the previous exercise to train a multi-class
%  classifier. 

%  Use lambda = 1e-4 for the weight regularization for softmax

% You need to compute softmaxModel using softmaxTrain on trainFeatures and
% trainLabels

inputSize  = 28 * 28;
numLabels  = numSelfTrainingDigits; 
hiddenSize = 200;
lambda = 1e-4;       % weight decay parameter       
maxIter = 400;

% the training based on extracted features, i.e. activation of hiddenlayer.
% the inputSize has been changed to hiddenSize
% digit 5-9 is used to train features.
% digit 0-4 is used to train softmax
% digit 0-4 is used to test the trained softmax
options.maxIter = maxIter;
softmaxModel = softmaxTrain(hiddenSize, numLabels, lambda, ...
                            trainFeatures, trainLabels, options);  

%% -----------------------------------------------------


%%======================================================================
%% STEP 5: Testing 

%% ----------------- YOUR CODE HERE ----------------------
% Compute Predictions on the test set (testFeatures) using softmaxPredict
% and softmaxModel

% You will have to implement softmaxPredict in softmaxPredict.m

% inputData is testFeatures
[pred] = softmaxPredict(softmaxModel, testFeatures);


%% -----------------------------------------------------

% Classification Score
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));

% (note that we shift the labels by 1, so that digit 0 now corresponds to
%  label 1)
%
% Accuracy is the proportion of correctly classified images
% The results for our implementation was:
%
% Accuracy: 98.3%
%
% Accuracy with 10 digits, only 96.520000%
