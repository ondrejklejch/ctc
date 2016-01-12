import sys
import time
import lasagne
import theano
import theano.tensor as T
import numpy as np
from networks import create_blstm_dropout
from ctc import compute_cost_log_in_parallel, LogSumExp
from load_data import load_data
from utils import evaluate_batch_on_data, viterbi_search_batch_compute

start = time.time()

theano.config.mode = 'FAST_RUN'
theano.config.floatX = 'float32'
np.random.seed(0)

batch_size = 64
num_inputs = 13
num_outputs = 62
hidden_layer_size = 256
lr = 0.025
dropout = 0.3
noise = 0.05
(train, validate, test) = load_data(batch_size)

input_vars = T.ftensor3('X')
mask_vars = T.matrix('mask')
target_vars = T.imatrix('y')
x_ends = T.ivector('x_ends')
y_ends = T.ivector('y_ends')
learning_rate = T.scalar('lr')

network = create_blstm_dropout(input_vars, mask_vars, num_inputs, hidden_layer_size, num_outputs, dropout, noise)
prediction = lasagne.layers.get_output(network, deterministic=True)
predict = theano.function([input_vars, mask_vars], prediction)

predictions = lasagne.layers.get_output(network, deterministic=False)
cost = T.sum(compute_cost_log_in_parallel(predictions, target_vars, LogSumExp, x_ends, y_ends))

params = lasagne.layers.get_all_params(network, trainable=True)
loss = T.mean(cost) + 1e-3 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
grad = T.grad(loss, params)
updates = lasagne.updates.nesterov_momentum(lasagne.updates.total_norm_constraint(grad, 1), params, learning_rate=learning_rate, momentum=0.9)

grad_norm = T.sum([T.sum(g**2) for g in lasagne.updates.total_norm_constraint(grad, 1)])
max_grad = T.max([T.max(T.abs_(g)) for g in lasagne.updates.total_norm_constraint(grad, 1)])
weight_norm = T.sum([T.sum(w**2) for w in params])

train_step = theano.function([input_vars, mask_vars, target_vars, x_ends, y_ends, learning_rate], (cost, predictions, grad_norm, weight_norm, max_grad), updates=updates, on_unused_input='ignore')

last_improvement = 1e10
epochs_since_last_improvement = 0
best_wer = 10
viterbi_search = viterbi_search_batch_compute()

print "Compiled, starting training"
for epoch in range(200):
	samples = range(len(train))
	np.random.shuffle(samples)

	epoch_training_start = time.time()
	total_cost = 0
	total_zero_predictions = 0
	for i in range(0, len(samples)):
		(x, y, x_ends, y_ends, mask) = train[samples[i]]

		cost, predictions, grad_norm, weight_norm, max_grad = train_step(x, mask, y, x_ends, y_ends, lr)

		if cost != cost:
			print "OVERFLOW"
			print "Total Cost: ", total_cost
			print "Actual Cost:", cost
			print "Zero predictions: ", total_zero_predictions
			print cost, grad_norm / weight_norm, grad_norm, weight_norm
			sys.exit(0)

		total_zero_predictions = np.sum(predictions == 0)
		total_cost += cost

	epoch_training_end = time.time()

	epoch_validation_start = time.time()
	stop, training_wer, learned = evaluate_batch_on_data(train, predict, viterbi_search)
	_, validation_wer, _ = evaluate_batch_on_data(validate, predict, viterbi_search)
	epoch_validation_end = time.time()

	best_wer = min(validation_wer, best_wer)

	print "\n\niteration: %d" % epoch
	print "training cost: " + str(total_cost)
	print "training cer: %.4f" % training_wer
	print "validation cer: %.4f (%.4f)" % (validation_wer, best_wer)
	print "epoch time: %.4f" % (epoch_training_end - epoch_training_start)
	if total_zero_predictions > 0:
		print "total zero predictions: %d" % total_zero_predictions
	check = [(validate[0][0][:5], validate[0][1][:5], validate[0][2][:5], validate[0][3][:5], validate[0][4][:5])]
	evaluate_batch_on_data(check, predict, viterbi_search, print_string=True)
	print >> sys.stderr, epoch, (time.time() - start), training_wer, validation_wer, (epoch_training_end - epoch_training_start), (epoch_validation_end - epoch_validation_start), weight_norm, lr

	if validation_wer < last_improvement:
		last_improvement = validation_wer
		epochs_since_last_improvement = 0
	else:
		epochs_since_last_improvement += 1

	if epochs_since_last_improvement > 10:
		print "DECREASING LR"
		lr /= 2.0
		epochs_since_last_improvement = 0

	if cost != cost:
		break


print "\n\n"
_, test_wer, _ = evaluate_batch_on_data(test, predict, viterbi_search, print_string = True)
print "\n\n"
print "Best validation cost:", best_wer
print "Test cost:", test_wer
