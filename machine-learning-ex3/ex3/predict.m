function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X = [ones(m, 1) X]; % Adding bias variable to features, n.
a1 = X'; % (n+1 X m) matrix. 
z2 = Theta1 * a1; % Transformed each sample from 401 features to 25 features
a2 = sigmoid(z2);
a2 = [ones(1, m); a2]; % Adding bias variable to features.
z3 = Theta2 * a2; % Transform each sample from 26 features to 10 features
a3 = sigmoid(z3);
h = a3';  % (m X 10)
[prob, idx] = max(h,[], 2);
p = idx;

% =========================================================================


end
