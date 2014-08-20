%%================================================================
clear ; close all; clc

addpath(genpath('../lib'));

%% Load data
%  Here we provide the code to load natural image data into x.
%  x will be a 144 * 10000 matrix, where the kth column x(:, k) corresponds to
%  the raw image data from the kth 12x12 image patch sampled.
%  You do not need to change the code below.

x = sampleIMAGESRAW();
figure('name','Raw images');
randsel = randi(size(x,2),200,1); % A random selection of samples for visualization
display_network(x(:,randsel));


%%================================================================
%% Test pca_whitening
%  Implement PCA with whitening and regularisation to produce the matrix
%  xPCAWhite. 

[xPCAWhite,U] = pca_whitening(x,1,0);
[xRegularizedPCAWhite] = pca_whitening(x,1,0.1);

%%================================================================
%% Check your implementation of PCA whitening 
%  Check your implementation of PCA whitening with and without regularisation. 
%  PCA whitening without regularisation results a covariance matrix 
%  that is equal to the identity matrix. PCA whitening with regularisation
%  results in a covariance matrix with diagonal entries starting close to 
%  1 and gradually becoming smaller. We will verify these properties here.
%  Write code to compute the covariance matrix, covar. 
%
%  Without regularisation (set epsilon to 0 or close to 0), 
%  when visualised as an image, you should see a red line across the
%  diagonal (one entries) against a blue background (zero entries).
%  With regularisation, you should see a red line that slowly turns
%  blue across the diagonal, corresponding to the one entries slowly
%  becoming smaller.

% -------------------- YOUR CODE HERE -------------------- 

covar = xPCAWhite*xPCAWhite'/size(xPCAWhite,2);
% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix without regularized whitening');
imagesc(covar);

reg_covar = xRegularizedPCAWhite*xRegularizedPCAWhite'/size(xRegularizedPCAWhite,2);
% Visualise the covariance matrix. You should see a red line across the
% diagonal against a blue background.
figure('name','Visualisation of covariance matrix with regularized whitening');
imagesc(reg_covar);

figure('name','PCA whitened images');
display_network(xPCAWhite(:,randsel));
%%================================================================
%% Implement ZCA whitening
%  Now implement ZCA whitening to produce the matrix xZCAWhite. 
%  Visualise the data and compare it to the raw data. You should observe
%  that whitening results in, among other things, enhanced edges.

% -------------------- YOUR CODE HERE -------------------- 
xZCAWhite = U*xPCAWhite;

% Visualise the data, and compare it to the raw data.
% You should observe that the whitened images have enhanced edges.
figure('name','ZCA whitened images');
display_network(xZCAWhite(:,randsel));

