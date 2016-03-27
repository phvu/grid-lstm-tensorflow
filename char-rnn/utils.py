import cPickle
import collections
import os

import numpy as np

class TextLoader(object):
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print "reading text file"
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print "loading preprocessed files"
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with open(input_file, "r") as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = list(zip(*count_pairs))
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'w') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(map(self.vocab.get, data))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file) as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)

    def create_batches(self):
        self.num_batches = self.tensor.size / (self.batch_size * self.seq_length)
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0


def visualize_result():
    import pandas as pd
    import matplotlib.pyplot as plt

    files = {'GridLSTM, 3 layers': 'save_gridlstm3layers/log.csv',
             'Stacked LSTM, 3 layers': 'save_lstm3layers/log.csv'}
    for k, v in files.iteritems():
        train_loss = pd.read_csv(v).groupby('epoch').mean()['train_loss']
        plt.plot(train_loss.index.tolist(), train_loss.tolist(), label=k)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Average training loss')
    plt.show()
