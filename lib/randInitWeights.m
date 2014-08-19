function theta = randInitWeights(L_out, L_in)
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the column row of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

%% Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(L_out+L_in+1);   % we'll choose weights uniformly from the interval [-r, r]
W = rand(L_out, L_in) * 2 * r - r;

b = zeros(L_out, 1);

% Convert weights and bias gradients to the vector form.
% This step will "unroll" (flatten and concatenate together) all 
% your parameters into a vector, which can then be used with minFunc. 
theta = [W(:) ; b(:) ];

end

