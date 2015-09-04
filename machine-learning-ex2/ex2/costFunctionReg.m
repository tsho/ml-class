function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

[J, grad] = costFunction(theta, X, y)

% fprintf('Theta(1): %f\n', theta(1));
% fprintf('Theta(2): %f\n', theta(2));
% fprintf('Theta(3): %f\n', theta(3));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
theta0 = [0; theta(2:end)];

h = sigmoid(X * theta);
J = J + (lambda / (2*m)) * sum(theta0 .^2)
grad = grad + (lambda / m) * theta0

% =============================================================

end
