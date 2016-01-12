import os
import numbers
import numpy as np
import tables
import theano

path = '/ha/work/people/klejch/saved_data.h5'

def load_data_timit():
    h5_file_path = os.path.join(path)
    h5_file = tables.openFile(h5_file_path, mode='r')

    def get_section(name):
        data_x = h5_file.get_node(h5_file.root, name + "_data_x")
        data_x_shapes = h5_file.get_node(h5_file.root, name + "_data_x_shapes")
        data_y = h5_file.get_node(h5_file.root, name + "_data_y")

        class _cVLArray(tables.VLArray):
            pass

        # A dirty hack to only monkeypatch data_x
        data_x.__class__ = _cVLArray

        # override getter so that it gets reshaped to 2D when fetched
        old_getter = data_x.__getitem__

        def getter(self, key):
            if isinstance(key, numbers.Integral) or isinstance(key, np.integer):
                return old_getter(key).reshape(data_x_shapes[key]).astype(theano.config.floatX)
            elif isinstance(key, slice):
                start, stop, step = self._processRange(key.start, key.stop, key.step)
                return [o.reshape(s) for o, s in zip(self.read(start, stop, step), data_x_shapes[slice(start, stop, step)])]

        # Patch __getitem__ in custom subclass, applying to all instances of it
        _cVLArray.__getitem__ = getter
        
        return (data_x, data_y)

    (train_x, train_y) = get_section("train")
    (valid_x, valid_y) = get_section("validate")
    (test_x, test_y) = get_section("test")
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    return rval

def load_data_timit_minibatches(batch_size = 10):
	(train, valid, test) = load_data_timit()

	x_mean = np.mean(np.concatenate(train[0]), axis=0)
	x_std = np.std(np.concatenate(train[0]), axis=0) 

	train_batches = make_batches(train[0], train[1], batch_size, x_mean, x_std)	
        valid_batches = make_batches(valid[0], valid[1], batch_size, x_mean, x_std)
	test_batches = make_batches(test[0], test[1], batch_size, x_mean, x_std)

	return (train_batches, valid_batches, test_batches)

def make_batches(x, y, batch_size, x_mean, x_std):
	batches = []
	sorted_by_length = sort_by_length(x,y)

	for i in range(0, len(x), batch_size):
		data = sorted_by_length[i:i+batch_size]	
		x_length = max([len(x) for (x,_) in data])

		xs = np.zeros((len(data), x_length, len(data[0][0][0])), dtype = np.float32)
		xs_lengths = np.zeros((len(data)), dtype = np.int32)
		mask = np.zeros((len(data), x_length), dtype=np.bool)
		for j in range(0, len(data)):
			x = data[j][0]
			xs[j, 0:len(x), 0:len(x[0])] = (x - x_mean) / x_std
			mask[j, 0:len(x)] = 1
			xs_lengths[j] = len(x)
		
		y_length = 2 * max([len(y) for (_,y) in data]) + 1
		ys = np.zeros((len(data), y_length), dtype = np.int32)
		ys_lengths = np.zeros((len(data)), dtype = np.int32)
		for j in range(0, len(data)): 
			y = interleave(data[j][1])
			ys[j, 0:len(y)] = y
			ys_lengths[j] = len(y)

		batches.append((preprocess(xs), ys, xs_lengths, ys_lengths, mask))

	return batches

def sort_by_length(x, y):
	sorted_indexes = sorted(range(len(x)), key=lambda index: len(x[index]))
	sorted_data = []
	for index in sorted_indexes:
		sorted_data.append((x[index], y[index]))

	return sorted_data

def interleave(y):
	interleaved = np.zeros(2 * y.shape[0] + 1, dtype=np.int32)
	interleaved[1::2] = y + 1

	return interleaved

