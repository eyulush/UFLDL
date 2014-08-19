function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% step 1 : calculate the target
z = theta * data; %numclass x inputsize * inputsize x numCases = numclass x numCases. note each column is an case
z = bsxfun(@minus,z,max(z,[],1));   % prevent the overflow. for each cases, substract the max number of class
z = exp(z);         
p = bsxfun(@rdivide, z, sum(z)); % calculate the target exp(theta*x) / sum(exp(theta*x))
        % sum function, is sum for each column. i.e. for each case

% step 2 : calculate the cost without regularization
Jcost = - sum(sum(groundTruth .* log(p))) / numCases; % groundTruth, 0-1 sparse matrix. size is numClasss x numCases

% step 3 : calculate the weight decay
Jweight = lambda * sum(sum(theta.^2)) / 2;

cost = Jcost + Jweight;

% step 4 : calculate the grad
Jgrad = -( (groundTruth - p) * data' ) / numCases; % numClasss x numCases * numCases x inputSize = numClass x inputSize

% step 5 : add regularity to grad
thetagrad = Jgrad + lambda*theta;
   
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];












% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

