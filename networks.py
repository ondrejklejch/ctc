import lasagne
import numpy as np
import theano.tensor as T
from lasagne.init import Constant, GlorotNormal, GlorotUniform, HeUniform
from lasagne.layers import InputLayer, GaussianNoiseLayer, RecurrentLayer, LSTMLayer, ReshapeLayer, DenseLayer, NonlinearityLayer, DropoutLayer, ElemwiseSumLayer
from lasagne.nonlinearities import sigmoid, rectify, leaky_rectify, softmax, identity, tanh


def create_dnn(input_vars, num_inputs, hidden_layer_size, num_outputs):
	network = InputLayer((None, None, num_inputs), input_vars)
	batch_size_theano, seqlen, _ = network.input_var.shape
	network = GaussianNoiseLayer(network, sigma=0.01)
	network = ReshapeLayer(network, (-1, 129))

	for i in range(1):
		network = DenseLayer(network, hidden_layer_size, W = GlorotUniform(), b = Constant(1.0), nonlinearity = leaky_rectify)

	network = ReshapeLayer(network, (-1, hidden_layer_size))
	network = DenseLayer(network, num_outputs, nonlinearity=softmax)
	network = ReshapeLayer(network, (batch_size_theano, seqlen, num_outputs))

	return network


def create_rnn(input_vars, num_inputs, hidden_layer_size, num_outputs):
	network = InputLayer((None, None, num_inputs), input_vars)
	batch_size_theano, seqlen, _ = network.input_var.shape
	network = GaussianNoiseLayer(network, sigma=0.05)

	for i in range(1):
		network = RecurrentLayer(network, hidden_layer_size, W_hid_to_hid = GlorotUniform(), W_in_to_hid = GlorotUniform(), b = Constant(1.0), nonlinearity = leaky_rectify, learn_init = True)

	network = ReshapeLayer(network, (-1, hidden_layer_size))
	network = DenseLayer(network, num_outputs, nonlinearity=softmax)
	network = ReshapeLayer(network, (batch_size_theano, seqlen, num_outputs))

	return network

def create_lstm(input_vars, num_inputs, hidden_layer_size, num_outputs):
	network = InputLayer((None, None, num_inputs), input_vars)
	batch_size_theano, seqlen, _ = network.input_var.shape
	network = GaussianNoiseLayer(network, sigma=0.01)

	for i in range(1):
		network = LSTMLayer(network, hidden_layer_size, learn_init = True)

	network = ReshapeLayer(network, (-1, hidden_layer_size))
	network = DenseLayer(network, num_outputs, nonlinearity=softmax)
	network = ReshapeLayer(network, (batch_size_theano, seqlen, num_outputs))

	return network

def create_blstm(input_vars, mask_vars, num_inputs, hidden_layer_size, num_outputs):
	network = InputLayer((None, None, num_inputs), input_vars)
	mask = InputLayer((None, None), mask_vars)
	batch_size_theano, seqlen, _ = network.input_var.shape
	network = GaussianNoiseLayer(network, sigma=0.01)

	for i in range(1):
		forward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True)
		backward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True, backwards = True)
		network =  ElemwiseSumLayer([forward, backward])

	network = ReshapeLayer(network, (-1, hidden_layer_size))
	network = DenseLayer(network, num_outputs, nonlinearity=softmax)
	network = ReshapeLayer(network, (batch_size_theano, seqlen, num_outputs))

	return network

def create_blstm_dropout(input_vars, mask_vars, num_inputs, hidden_layer_size, num_outputs, dropout = 0.2, noise = 0.2):
	network = InputLayer((None, None, num_inputs), input_vars)
	mask = InputLayer((None, None), mask_vars)
	batch_size_theano, seqlen, _ = network.input_var.shape
	network = GaussianNoiseLayer(network, sigma=noise)

	for i in range(4):
		forward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True)
		backward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True, backwards = True)
		network = DropoutLayer(GaussianNoiseLayer(ElemwiseSumLayer([forward, backward]), noise), dropout)

	network = ReshapeLayer(network, (-1, hidden_layer_size))
	network = DenseLayer(network, num_outputs, nonlinearity=softmax)
	network = ReshapeLayer(network, (batch_size_theano, seqlen, num_outputs))

	return network

def create_full_blstm_dropout(input_vars, mask_vars, num_inputs, hidden_layer_size, num_outputs, dropout = 0.2, noise = 0.2):
	network = InputLayer((None, None, num_inputs), input_vars)
	mask = InputLayer((None, None), mask_vars)
	batch_size_theano, seqlen, _ = network.input_var.shape
	network = GaussianNoiseLayer(network, sigma=noise)

	network = ReshapeLayer(network, (-1, num_inputs))
	network = DenseLayer(network, hidden_layer_size, nonlinearity=tanh)
	network = ReshapeLayer(network, (batch_size_theano, seqlen, hidden_layer_size))

	layers = [network]

	for i in range(4):
		network = ElemwiseSumLayer(layers)

		forward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True)
		backward = LSTMLayer(network, hidden_layer_size, mask_input = mask, learn_init = True, backwards = True)
		network = DropoutLayer(GaussianNoiseLayer(ElemwiseSumLayer([forward, backward]), noise), dropout)

		layers.append(network)

	network = ReshapeLayer(network, (-1, hidden_layer_size))
	network = DenseLayer(network, num_outputs, nonlinearity=softmax)
	network = ReshapeLayer(network, (batch_size_theano, seqlen, num_outputs))

	return network
