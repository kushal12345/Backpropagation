import numpy as np

def sigmoid(x):
    x=1/(1+np.exp(-x))
    return x
    pass

def derivative(a):
    return a*(1-a)

x = np.array([0.5, 0.1, -0.2])
target = 0.6
learning_rate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

# forward pass


hidden_layer_input = np.dot(x,weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output,weights_hidden_output.T)
output = sigmoid(output_layer_in)
error = target - output
print(error)

print('Error: ' + str(np.mean(np.abs(error))))
#backwardpass

del_err_output=error * derivative(output)
error_first_layer=np.dot(del_err_output, weights_hidden_output)

del_err_input=error_first_layer * derivative(hidden_layer_output)

delta_w_h_o = learning_rate * del_err_output * hidden_layer_output
delta_w_i_o = learning_rate * del_err_input * x[:, None]

print('Change in weights for hidden layer to output layer:')
print(hidden_layer_output)
print('Change in weights for input layer to hidden layer:')
print(hidden_layer_input)
