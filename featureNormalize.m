function [X_norm, mu, sigma] = featureNormalize(X)
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2)); 

mu = mean(X);
sigma = std(X);

iterations = 1:size(X,2);
for i = iterations
  Xminusmu = X(:,i) - mu(i);
  X_norm(:,i) = Xminusmu / sigma(i);
end

% ============================================================

end
