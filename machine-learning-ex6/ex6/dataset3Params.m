function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
accuracy = 0;
steps = [0.01;0.03;0.1;0.3;1;3;10;30]
for c = 1:size(steps,1)
  for sig = 1: size(steps,1)
    model= svmTrain(X, y, steps(c,1), @(x1, x2) gaussianKernel(x1, x2, steps(sig,1)));
    prediction = svmPredict(model, Xval);
    ai = mean(double(prediction ~= yval));
    fprintf("sigma= %d, C= %d, accuracy= %d \n", steps(sig,1), steps(c,1), ai);
    if ai > accuracy
      accuracy = ai;
      C = steps(c,1);
      sigma = steps(sig,1);
    end 
  end
end

% =========================================================================

end
