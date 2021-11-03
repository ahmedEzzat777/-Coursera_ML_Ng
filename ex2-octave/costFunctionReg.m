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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta); % m * 1

Jpos = -log(h);  % m * 1
Jneg = -log(1-h); % m * 1
modTheta = theta;
modTheta(1) = 0;
J = (Jpos'*y + Jneg'*(1-y))/m + (lambda*(modTheta' * modTheta))/(2*m); % (1* m) * (m*1)

grad = (X' * (h - y))/m + (lambda*modTheta)/m; % (n+1 * m) * (m * 1)



% =============================================================

end
