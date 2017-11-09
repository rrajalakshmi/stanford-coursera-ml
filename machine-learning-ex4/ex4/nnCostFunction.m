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
% Take 1: 25* 401 elements as Theta1 25 X 401 matrix
% Take (25*401+1): end and transform to Theta2 10 X 26 matrix
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

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
a1 = [ ones(m,1) X]; 
z2 = a1 * Theta1' ;
a2 = sigmoid(z2);
z3 = [ones(m,1) a2] * Theta2';
h = sigmoid(z3);
c = 0;
for i = 1:m
    yk = zeros(1, num_labels);
    yk(1,y(i)) = 1;
    c = c + -yk * log(h(i,:))' + -(1-yk) * log(1-h(i,:))';
end

J = c / m;
% sum function sums up all the rows. When applied to a vector, sums up all columns
R = (sum(sum(Theta1(:,2:end) .^ 2)) + sum(sum(Theta2(:,2:end) .^ 2)));
R = R*lambda/(2*m);
J = J + R;

%back propogation
T = 0;

a1 = [ones(1,m); X']; 
for i = 1:m
  %layer 1-2
  
  ai1 = a1(:,i); % 401 X 1
  zi2 = Theta1 * ai1; % (25 X 401) * (401 X 1) = (25 X 1)
  ai2 = sigmoid(zi2); % 25 X 1
  % layer 2-3
  ai2 = [ones(1,1); ai2]; 
  zi3 =  Theta2 * ai2; % (10 X 26) * (26 X 1)
  hi = sigmoid(zi3);
  yk = ([1:num_labels]==y(i))'; %(10 X1)
  e3 = hi - yk; % 10 X 1
  e2 = (Theta2' * e3) .* [ones(1,1); sigmoidGradient(zi2)]; %(26X10) * (10X1) .* (26X1) = (26X1)
  %remove the bias. We get one error term fo each unit in layer 2.  
  e2 = e2(2:end,:); %(25X1)
  Theta2_grad = Theta2_grad + e3 * ai2'; %(10X26)
  Theta1_grad = Theta1_grad + e2 * ai1'; %(25 X 401)    
end

%Theta1_grad = (1/m).* Theta1_grad;
%Theta2_grad = (1/m) .* Theta2_grad;

%Theta_grad(i,j) represent the derivative of the cost function over Theta(i,j)
% Do not regularize the bias column
Theta1_grad = (1/m).* Theta1_grad + (lambda/m) .* [zeros(size(Theta1,1),1) Theta1(:,2:end)];
Theta2_grad = (1/m) .* Theta2_grad + (lambda/m) .* [zeros(size(Theta2,1),1) Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
