function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% step 0 : preparation, transform the data to be 1 column as 1 training
% example
m = size(data,2); 
if m == visibleSize
    data = data';
    m = size(data,2);
end

% step 1 : feed forward 
a1 = [ones(1,m);data];  % adding 1 to each training example for bais for input layer a1, e.g. 65x10000
z2 = [b1,W1]*a1;         % size example, 25*65 x 65*10000 = 25*10000
a2 = [ones(1,m);sigmoid(z2)]; % adding 1 to each training example for bais for hidden layer a1, e.g. 26x10000
z3 = [b2,W2]*a2;         % W2 64x25, z2 size 64*26 x 26*10000 = 64*10000
a3 = sigmoid(z3);       % a3 is output, size 64*10000

% step 2 : calculate the cost, squared error for autoencoder, 
% without regularization
cost = sum(sum((data-a3).^2)) / (2*m);

% step 3 : calculate regularization
Jweight =   (sum(sum(W1.^2)) + sum(sum(W2.^2))) / 2;

% step 4 : calculate the cost with sparsity penalty
rho = mean(a2(2:end,:),2);    % average activity for hidden unit 25x1
Jsparse = sum(sparsityParam*log(sparsityParam./rho) + (1-sparsityParam)*log((1-sparsityParam)./(1-rho)));

% use ./ for vector operation 
% and then sum for all hidden unit
% cost = cost + lambda *Jweight;
cost = cost + lambda *Jweight + beta*Jsparse;

% step 5 : calculate the graident
% note : define the delta as gradient of cost to z, i.e. dE/dz
% if total  layer is n, a(n) is output
% delta(n)     =  (a(n)-y) * a(n) * (1-a(n))
% delta(n-1) = delta(n) * theta(n-1) * a(n-1) * (1-a(n-1))
% And sigmoidGradient(z(n)) = a(n) * (1-a(n))

% ==> delta(n)     = (a(n)-y) .* sigmoidGradient(z(n))
% ==> delta(n-1) = delta(n) * theta(n-1) * sigmoidGradient(z(n-1))
% ==> delta(n-2) = delta(n-1) * theta(n-2) .* sigmoidGradient(z(n-2))

delta_3 = (a3-data) .* sigmoidGradient(z3) ;      % 64 x 10000
delta_2 =  ([b2,W2]'*delta_3);      % 26*64 x 64*10000 = 26 x 10000
delta_2 = delta_2(2:end,:);          % 25x10000

% for delta_2, i.e. hidden layer, add sparisity penalty, so delta_2 need to
% be changed as
gSparse = beta * (-sparsityParam./rho + (1-sparsityParam) ./ (1-rho) ) ; % size 25x1
delta_2 = (delta_2 + repmat(gSparse,1,m)) .* sigmoidGradient(z2); 

Delta_2 = (a2 * delta_3')'; % size 26x64 the 1st column is for b2
Delta_1 = (a1 * delta_2')'; % size 65x25 the 1st column is for b1

W2grad = Delta_2 / m; % the sum function already in matrix multiply
W1grad = Delta_1 / m;

b2grad = W2grad(:,1);
b1grad = W1grad(:,1);
W2grad = W2grad(:,2:end);
W1grad = W1grad(:,2:end);

% step 6: calculate the regaulized gradient 
W2grad = W2grad + lambda * W2;
W1grad = W1grad + lambda * W1;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.
%  g'(z) = g(z)*(1-g(z))

g = sigmoid(z).*(ones(size(z))-sigmoid(z));
end

