function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer1_size, ...
                                   num_labels, ...
                                   X_norm, y, lambda)

shape1 = hidden_layer_size * (input_layer_size + 1);
shape2 = hidden_layer1_size*(hidden_layer_size+1);
                                   
Theta1 = reshape(nn_params(1:shape1), ...
                 hidden_layer_size, (input_layer_size + 1));
                 
Theta2 = reshape(nn_params(1+shape1:shape1+shape2),...
                 hidden_layer1_size,(hidden_layer_size+1));
                 
Theta3 = reshape(nn_params(1+shape1+shape2:end),...
                 num_labels,(hidden_layer1_size+1));
                 

size_t1 = size(Theta1);
size_t2 = size(Theta2);
size_t3 = size(Theta3);
%Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):(hidden_layer1_size+), ...
 %                hidden_layer1_size, (hidden_layer_size + 1));
                 
%Theta3 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                 num_labels, (hidden_layer1_size + 1));
                 

m = size(X_norm, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

size_xnorm = size(X_norm);
a1 = [ones(m,1) X_norm];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
a3 = [ones(m,1) a3];
z4 = a3* Theta3';
h_x = sigmoid(z4);

Y = zeros(m,num_labels);
for i = 1:m
  num = y(i,1); 
  Y(i,y(i,1)) = 1;
endfor

size_h_x = size(h_x);
size_y = size(Y);
J = (1/m)*sum(sum((-Y.*log(h_x)) - ((1-Y).*log(1-h_x))));

Theta_array = {Theta1;Theta2;Theta3};
final_sum = 0;
for j = 1:size(Theta_array,1)
  temp_theta = zeros(size(Theta_array,j));
  temp_theta = Theta_array{j,1};
  temp_sum = sum(sum(temp_theta(:,2:end).^2));
  final_sum = final_sum + temp_sum;
endfor

reg_param = lambda/(2*m) * final_sum;
J = J + reg_param;

for u = 1:m 
  a1 = [1 ; X_norm(u,:)'];
  z2 = Theta1 * a1;
  temp_a2 = sigmoid(z2);
  a2 = [1; temp_a2];
  z3 = Theta2 * a2;
  temp_a3 = sigmoid(z3);
  a3 = [1; temp_a3];
  z4 = Theta3 * a3;
  a4 = sigmoid(z4);
  sub_Y = (Y(u,:))';
  delta_four = a4 - sub_Y;
  g_prime_z3 = a3.*(1-a3);
  delta_three = (Theta3'*delta_four).*[g_prime_z3];
  delta_three = delta_three(2:end);
  g_prime_z2 = a2.*(1-a2);
  delta_two = (Theta2'*delta_three).*[g_prime_z2];
 delta_two = delta_two(2:end);
 Theta1_grad = Theta1_grad + delta_two*a1';
 Theta2_grad = Theta2_grad + delta_three * a2';
 Theta3_grad = Theta3_grad + delta_four * a3';
endfor

Theta1_grad = (1/m) * [(Theta1_grad) + (lambda * [zeros(size(Theta1,1),1) Theta1(:,2:end)])];
Theta2_grad = (1/m) * [(Theta2_grad) + (lambda * [zeros(size(Theta2,1),1) Theta2(:,2:end )])];
Theta3_grad = (1/m) * [(Theta3_grad) + (lambda * [zeros(size(Theta3,1),1) Theta3(:,2:end )])];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];
end
