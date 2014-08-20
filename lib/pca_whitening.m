function [xPCAWhite, U, k] = pca_whitening(x, k_threshold, epsilon)
% This function is doing pca whitening for the data
% Note : the input data will be transformed with zero-mean
% Note : the input data should be elemSize x elemNum, i.e. 1 column as 1 case
% Input : 
%   x : the data need to be whitened
%   k_threshold : how much variance will be retained. for example 0.99
%   epsilon : the regularized parameter. for example 1e-4
% Output :
%   xPCAWhite : the data after whitening
%   U : the eigenvector
% Note :
%   ZCA whitening result will be U*xPCAWhite

warning off all

% Zero-mean the data (by row) Example
avg = mean(x,1);
x = x - repmat(avg,size(x,1),1);

% Compute the eigenvector and eigenvalue by svd
sigma = x*x'/size(x,2);
[U,S,V] = svd(sigma);

%  Find k, the number of components to retain
%  Write code to determine k, the number of components to retain in order
%  to retain at least 99% of the variance.

lambda = max(S,[],2);   % 1 column vector
lambda_sum = sum(lambda);

if k_threshold >= 1
    k = size(U,2);
else
    lambda_accu = 0;
    for k = 1:length(lambda)
        lambda_accu = lambda_accu + lambda(k);
        if lambda_accu / lambda_sum >= k_threshold
            break;
        end
    end
end

% Implement PCA with dimension reduction
xTilde = U(:,1:k)' * x;
xHat = U(:,1:k)*xTilde; % restore the xHat from xTilde, UU' = I, xHat is on lower dimesion reduction

% PCA whitening 
xPCAWhite = xTilde ./ repmat(sqrt(lambda+epsilon),1,size(xTilde,2));

warning on all

