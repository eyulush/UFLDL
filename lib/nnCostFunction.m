function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % size 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % size 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Step 1, calculate the h_theta(x), i.e. a3 and transform the y from vector to matrix m*10
a1 = [ones(m,1),X]; % size 5000x401
z2 = a1*Theta1';     % size 5000x25
a2 = [ones(m,1),sigmoid(z2)]; % size 5000x26
z3 = a2*Theta2';     % size 5000x10
a3 = sigmoid(z3);   % size 5000x10

% expand y = [ 1; 2; 3; ... ] to [ [ 1, 0, 0, ...] ; [ 0, 1, 0, ...]; ... ]  
v_y = zeros(m,num_labels);
v_y(sub2ind(size(v_y),[1:m],y')) = 1;

% Step 2, calculate the cost with regularization, 2 sum functions needed
J = sum(sum ( -v_y.*log(a3) - (ones(size(v_y)) - v_y) .* log(ones(size(a3)) - a3))) / m;

% Step 3, calculate with regularization, size(Theta1)=25x401, only 25x400 needed for regularization, same for Theta2
J = J + lambda * (sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2))) / (2*m);

% Step 4, calculate the gradient
% Note : for the output layer, dE/dz3 = dE/da3 * da3/dz3 = (-y / a3 -
% (1-y)*-1/(1-a3) ) * (a3 * (1-a3)) = a3-y
% because we use logistic error not square error. so delta_3 = a3-y
% if we use square error, delta_3 = (a3-y) * sigmodGradient(z3) 
delta_3 = (a3 - v_y); % size 5000x10
delta_2 = delta_3*Theta2; % size 5000x26
delta_2 = delta_2(:,2:end); % size 5000x25
delta_2 = delta_2 .* sigmoidGradient(z2); % size 5000x25 .* 5000x25 = 5000x25

Delta_2 = delta_3' * a2; % size 10x26
Delta_1 = delta_2' * a1; % size 25x401

Theta2_grad = Delta_2 / m;
Theta1_grad = Delta_1 / m;

% Step 5, calculate the regaulized gradient
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / (m) * Theta2(:,2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / (m) * Theta1(:,2:end);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
