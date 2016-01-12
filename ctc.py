import numpy as np
import theano
import theano.tensor as T
from lasagne.objectives import categorical_crossentropy, aggregate, squared_error

def LogSumExp(x, axis=None):
    x = T.clip(x, -1e20, 1e20)
    x_max = T.max(x, axis=axis)
    return T.log(T.sum(T.exp(x - x_max), axis=axis)) + x_max

def shift_matrix(matrix, shift):
	new_matrix = T.zeros_like(matrix)
	return T.set_subtensor(new_matrix[:,shift:], matrix[:,:-shift])

def log_shift_matrix(matrix, shift):
	new_matrix = T.log(T.zeros_like(matrix))
	return T.set_subtensor(new_matrix[:,shift:], matrix[:,:-shift])

def compute_cost_log_in_parallel(original_rnn_outputs, labels, func, x_ends, y_ends):
	mask = 1 - T.or_(T.eq(labels, T.zeros_like(labels)), T.eq(labels, shift_matrix(labels, 2)))

	initial_state = T.zeros_like(labels)
	initial_state = T.set_subtensor(initial_state[:,0], 1)

	def select_probabilities(rnn_outputs, label):
		return rnn_outputs[:,label]	

	rnn_outputs, _ = theano.map(select_probabilities, [original_rnn_outputs, labels])
	rnn_outputs = rnn_outputs.dimshuffle((1,0,2))

	def forward_step(probabilities, last_probabilities):
		all_forward_probabilities = T.stack(
			last_probabilities * probabilities,
			shift_matrix(last_probabilities, 1) * probabilities,
			shift_matrix(last_probabilities, 2) * probabilities * mask,
		)

		result = func(all_forward_probabilities, 0)
		return result

	forward_probabilities, _ = theano.scan(fn = forward_step, sequences = rnn_outputs, outputs_info = initial_state)
	forward_probabilities = forward_probabilities.dimshuffle((1,0,2))

	def compute_cost(forward_probabilities, x_end, y_end):
		return -T.log(func(forward_probabilities[x_end-1,y_end-2:y_end]))

	return theano.map(compute_cost, [forward_probabilities, x_ends, y_ends])[0]

def compute_cost_log_in_parallel(original_rnn_outputs, labels, func, x_ends, y_ends):
	mask = T.log(1 - T.or_(T.eq(labels, T.zeros_like(labels)), T.eq(labels, shift_matrix(labels, 2))))

	initial_state = T.log(T.zeros_like(labels))
	initial_state = T.set_subtensor(initial_state[:,0], 0)

	def select_probabilities(rnn_outputs, label):
		return rnn_outputs[:,label]	

	rnn_outputs, _ = theano.map(select_probabilities, [original_rnn_outputs, labels])
	rnn_outputs = T.log(rnn_outputs.dimshuffle((1,0,2)))

	def forward_step(probabilities, last_probabilities):
		all_forward_probabilities = T.stack(
			last_probabilities + probabilities,
			log_shift_matrix(last_probabilities, 1) + probabilities,
			log_shift_matrix(last_probabilities, 2) + probabilities + mask,
		)

		result = func(all_forward_probabilities, 0)
		return result

	forward_probabilities, _ = theano.scan(fn = forward_step, sequences = rnn_outputs, outputs_info = initial_state)
	forward_probabilities = forward_probabilities.dimshuffle((1,0,2))

	def compute_cost(forward_probabilities, x_end, y_end):
		return -func(forward_probabilities[x_end-1,y_end-2:y_end])

	return theano.map(compute_cost, [forward_probabilities, x_ends, y_ends])[0]

def compute_cost_with_cross_entropy_in_parallel(original_rnn_outputs, labels, x_ends, y_ends):
	mask = T.log(1 - T.or_(T.eq(labels, T.zeros_like(labels)), T.eq(labels, shift_matrix(labels, 2))))
	arange = T.arange(labels.shape[1])

	initial_state = T.log(T.zeros_like(labels))
	initial_state = T.set_subtensor(initial_state[:,0], 0)

	def select_probabilities(rnn_outputs, label):
		return rnn_outputs[:,label]	

	rnn_outputs, _ = theano.map(select_probabilities, [original_rnn_outputs, labels])
	rnn_outputs = T.log(rnn_outputs.dimshuffle((1,0,2)))

	def forward_step(probabilities, last_probabilities):
		all_forward_probabilities = T.stack(
			last_probabilities + probabilities,
			log_shift_matrix(last_probabilities, 1) + probabilities,
			log_shift_matrix(last_probabilities, 2) + probabilities + mask,
		)

		max_probability, backlink = T.max_and_argmax(all_forward_probabilities, 0)
		backlink = arange - backlink
		return max_probability, backlink

	results, _ = theano.scan(fn = forward_step, sequences = rnn_outputs, outputs_info = [initial_state, None])
	forward_probabilities, backward_pointers = results

	def compute_cost(rnn_outputs, forward_probabilities, backward_pointers, x_end, y_end, label):
		def backward_step(backlinks, position):
			new_position = backlinks[position]
			return new_position, position

		initial_state = T.argmax(forward_probabilities[x_end-1,y_end-2:y_end]) + y_end - 2

		results, _ = theano.scan(fn = backward_step, sequences = backward_pointers[0:x_end,:], outputs_info = [initial_state, None], go_backwards = True)
		alignment = label[results[1][::-1]]

		return aggregate(categorical_crossentropy(rnn_outputs[0:x_end], alignment), mode='sum')

	forward_probabilities = forward_probabilities.dimshuffle((1,0,2))
	backward_pointers = backward_pointers.dimshuffle((1,0,2))

	return theano.map(compute_cost, [original_rnn_outputs, forward_probabilities, backward_pointers, x_ends, y_ends, labels])[0]
