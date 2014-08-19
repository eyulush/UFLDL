function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

z = theta * data; %numclass x inputsize * inputsize x numCases = numclass x numCases. note each column is an case
% z = bsxfun(@minus,z,max(z,[],1));   % prevent the overflow. for each cases, substract the max number of class
% z = exp(z);         
% p = bsxfun(@rdivide, z, sum(z)); % calculate the target exp(theta*x) / sum(exp(theta*x))
        % sum function, is sum for each column. i.e. for each case

[r,c] = find(bsxfun(@eq,z,max(z,[],1)));
pred = r';



% ---------------------------------------------------------------------

end

