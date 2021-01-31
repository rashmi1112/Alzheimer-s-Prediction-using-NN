function p = NNpredict(Theta1, Theta2, Theta3, X)
m = size(X, 1);
num_labels = size(Theta3, 1);

p = zeros(size(X, 1), 1);


for i = 1:size(X,1)
  a1 = [1 X(i,:)];
  size_a1 = size(a1);
  size_theta1 = size(Theta1');
  z2 = a1 * Theta1';
  a2 = sigmoid(z2);
  a2 = [1 a2];
  z3 = a2 * Theta2';
  size_z3 = size(z3);
  a3 = sigmoid(z3);
  a3 = [1 a3];
  z4 = a3 * Theta3';
  a4 = sigmoid(z4);
  [val, idx] = max(a4);
  p(i,1) = idx;
end
end
