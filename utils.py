import theano
import theano.tensor as T
import numpy as np
from phones import intToPhones, intToReducedPhones

def decode_to_string(encoded_string):
	string = []
	last = 0
	for c in encoded_string:
		if c != last and c != 0 and (c-1) in intToReducedPhones:
			string.append(intToReducedPhones[c-1])
			last = c

	return ' '.join(string)

def evaluate_batch_on_data(data, predict, viterbi_search, print_string = False):
	stop = True
	wer_errors = 0.0
	wer_length = 0
	learned = 0

	for batch_index in range(len(data)):
		(x, y, x_ends, y_ends, mask) = data[batch_index]
		rnn_outputs = predict(x, mask)

		search = viterbi_search(rnn_outputs)
		for i in range(len(x)):
			original = decode_to_string(y[i][0:y_ends[i]])
			decoded = decode_to_string(search[i][0:x_ends[i]])

			wer_errors += wer(original.split(" "), decoded.split(" "))
			wer_length += len(original.split(" "))

			if print_string:
				print "Expected: %s" % original
				print "Received: %s" % decoded

			if original == decoded:
				learned += 1
			else:
				stop = False

	return stop, (wer_errors / wer_length), learned

def viterbi_search_batch_compute():
	rnn_outputs = T.ftensor3()
	return theano.function([rnn_outputs], T.argmax(rnn_outputs, 2))

def wer(r, h):
    """
        Calculation of WER with Levenshtein distance.
        Works only for iterables up to 254 elements (uint8).
        O(nm) time ans space complexity.

        >>> wer("who is there".split(), "is there".split()) 
        1
        >>> wer("who is there".split(), "".split()) 
        3
        >>> wer("".split(), "who is there".split()) 
        3
    """
    # initialisation
    import numpy
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]
