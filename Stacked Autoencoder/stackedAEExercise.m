%% CS294A/CS294W Stacked Autoencoder Exercise

% eyulush : adding some debug, trace control
clear ; close all; clc
addpath(genpath('../lib'));

debug = 1;
currPhrase = 2;

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 28 * 28;
numClasses = 10;
hiddenSizeL1 = 200;    % Layer 1 Hidden Size
hiddenSizeL2 = 200;    % Layer 2 Hidden Size
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term      
trainingSize = 60000;

sae1IterStep = 20;
sae2IterStep = 100;

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
trainData = loadMNISTImages('mnist/train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels.idx1-ubyte');

trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1

testData = loadMNISTImages('mnist/t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('mnist/t10k-labels.idx1-ubyte');

testLabels(testLabels == 0) = 10; % Remap 0 to 10

if trainingSize < size(trainData,2)
    trainData = trainData(:,1:trainingSize);
    trainLabels = trainLabels(1:trainingSize,:);
end

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.


%  Randomly initialize the parameters
if exist('saves\dnn_parameters.mat','file')  == 2
    % load parameter, including
    % hiddenSizeL1,hiddenSizeL2, currIter, aeTheta1
    load('saves/dnn_parameters.mat');
else
    sae1OptTheta = initializeParameters(hiddenSizeL1, inputSize);
    sae2OptTheta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
    saeSoftmaxOptTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
    save('saves/dnn_parameters.mat', 'sae1OptTheta','sae2OptTheta','saeSoftmaxOptTheta');
    
    currSae1Iter = 0;
    currSae2Iter = 0;
    save('saves/dnn_parameters.mat', 'currSae1Iter','currSae2Iter','-append');
end

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta

if currPhrase == 1          % training the 1st hidden layer autoencoder
    %  Use minFunc to minimize the function
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
    options.maxIter = sae1IterStep;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';

    % use trainData for self-unsupervised-training
    [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                                   sae1OptTheta, options);

    % update sae1OptTheta
    currSae1Iter = currSae1Iter + sae1IterStep;
    save('saves/dnn_parameters.mat', 'sae1OptTheta','currSae1Iter','-append');


    % Visualize weights
    W1 = reshape(sae1OptTheta(1:hiddenSizeL1 * inputSize), hiddenSizeL1, inputSize);
    display_network(W1');
    
    % Extrace 1st features
    [sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);
    [testFeaturesL1] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, testData);
    save('saves/dnn_parameters.mat', 'sae1Features', 'testFeaturesL1','-append');
                                    
end
% -------------------------------------------------------------------------


%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.



%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

if currPhrase == 2      % training 2nd hidden layer autoencoder
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
    options.maxIter = sae2IterStep;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';

    [sae2OptTheta, cost] =  minFunc(@(p)sparseAutoencoderCost(p,...
        hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),sae2OptTheta,options);

    currSae2Iter = currSae2Iter + sae2IterStep;
    save('saves/dnn_parameters.mat', 'sae2OptTheta','currSae2Iter','-append');
    
    [sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);
    
    [testFeaturesL2] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, testFeaturesL1);

	save('saves/dnn_parameters.mat', 'sae2Features', 'testFeaturesL2','-append');
end

% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.



%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);
if currPhrase == 1
    trainFeatures = sae1Features;
    testFeatures = testFeaturesL1;
end
if currPhrase == 2 || currPhrase == 3
    trainFeatures = sae2Features;
    testFeatures = testFeaturesL2;
end

if currPhrase < 4
    options.Method = 'lbfgs';
    options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';

    [saeSoftmaxOptTheta, cost] = minFunc( @(p) softmaxCost(p, ...
                                   numClasses, hiddenSizeL2, lambda, ...
                                   trainFeatures, trainLabels), ...                                   
                                   saeSoftmaxOptTheta, options);
        
    save('saves/dnn_parameters.mat', 'saeSoftmaxOptTheta','-append');

    saeSoftmaxOptTheta = reshape(saeSoftmaxOptTheta(1:numClasses * hiddenSizeL2), numClasses, hiddenSizeL2);
    softmaxModel.optTheta = saeSoftmaxOptTheta;
    [pred] = softmaxPredict(softmaxModel, testFeatures);    
    if debug == 1
        testAccurancy = 100*mean(pred(:) == testLabels(:)); 
        
        fprintf('Test Accuracy: %f%%\n', testAccurancy);
        save('saves/dnn_parameters.mat','testAccurancy','-append');
        return
    end
end

%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%
options = struct;
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL2,...
                         numClasses, netconfig,lambda, trainData, trainLabels),...
                        stackedAETheta,options);
save('saves/stackedAEOptTheta.mat', 'stackedAEOptTheta');

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
% Get test data part has been moved to beginning

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

acc = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
