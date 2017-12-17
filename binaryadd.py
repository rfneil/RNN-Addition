import copy
import numpy as np
np.random.seed(1)

def sigmoid(x):
		output = 1/(1 + np.exp(-x))
		return output

def sigmoid_derivative(x):
		derivative = x*(1-x)
		return derivative

#dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
		int2binary[i] = binary[i]



#input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1

#initialize weights between (-1,1)
w0 = 2*np.random.random((input_dim,hidden_dim)) - 1
w1 = 2*np.random.random((hidden_dim,output_dim)) - 1
w2 = 2*np.random.random((hidden_dim,hidden_dim)) - 1

dw0 = np.zeros_like(w0)
dw1 = np.zeros_like(w1)
dw2 = np.zeros_like(w2)

#training
for j in range(10000):

	#generate a simple addition a + b = c
	a_int = np.random.randint(largest_number/2)
	a = int2binary[a_int]

	b_int = np.random.randint(largest_number/2)
	b = int2binary[b_int]

	#true value
	c_int = a_int+b_int
	c = int2binary[c_int]

	#store the guess
	d = np.zeros_like(c)

	error = 0

	layer_2_deltas = list()
	layer_1_values = list()
	layer_1_values.append(np.zeros(hidden_dim))

	#moving along the binary coding

	for p in range(binary_dim):

		#generate input and output
		X = np.array([[a[binary_dim - p -1] ,b[binary_dim - p -1]]])
		y = np.array([[c[binary_dim - p -1]]]).T

		#hidden layer = input + prev_hidden
		layer_1 = sigmoid(np.dot(X,w0) + np.dot(layer_1_values[-1],w2))

		#output layer
		layer_2 = sigmoid(np.dot(layer_1, w1))

		#backprop
		layer_2_error = y - layer_2
		layer_2_deltas.append((layer_2_error)*sigmoid_derivative(layer_2))
		error += np.abs(layer_2_error[0])

		#estimate value
		d[binary_dim - p - 1] = np.round(layer_2[0][0])

		#store hidden layer to use it in the next time step
		layer_1_values.append(copy.deepcopy(layer_1))

	future_layer_1_delta = np.zeros(hidden_dim)

	for p in range(binary_dim):

		X = np.array([[a[p],b[p]]])
		layer_1 = layer_1_values[-p-1]
		prev_layer_1 = layer_1_values[-p-2]

		#error at output layer
		layer_2_delta = layer_2_deltas[-p-1]

		#error at hidden layer
		layer_1_delta = (future_layer_1_delta.dot(w2.T) + layer_2_delta.dot(w1.T)) * sigmoid_derivative(layer_1)

		#weight update
		dw1 += np.atleast_2d(layer_1).T.dot(layer_2_delta)
		dw2 += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
		dw0 += X.T.dot(layer_1_delta)

		future_layer_1_delta = layer_1_delta

	w0 += dw0 * alpha
	w1 += dw1 * alpha
	w2 += dw2 * alpha

	dw0 *=0
	dw1 *=0
	dw2 *=0

	#print progress
	if(j%1000 ==0):
		print("Error : " + str(error))
		print("Prediction : "+ str(d))
		print("True Value : " + str(c))

		out = 0

		for index,x in enumerate(reversed(d)):
			out += x*pow(2,index)
		print(str(a_int)+ " + " + str(b_int)+ " = " + str(out))
		print("--------")








