#char-rnn-gridlstm

[GridLSTM](http://arxiv.org/abs/1507.01526) (and GridRNN in general) for character-level language models in 
Python using Tensorflow.

This serves as a simple test for GridLSTM in tensorflow. The rest of the code 
(minus some small changes in `model.py`) come from 
[sherjilozair's character level rnn repo](https://github.com/sherjilozair/char-rnn-tensorflow).

Basically you will only need to run the following command:

    $ python train.py

Note that the GridRNN implementation in tensorflow is generic, in the sense that it supports multiple 
dimensions with various settings for input/output dimensions, priority dimensions and non-recurrent dimensions.
The type of recurrent cell can also be selected among LSTM, GRU or vanilla RNN.

For the purpose of character-level language modeling, however, we only use 2GridLSTM here.
More test cases for GridLSTM will be added later.
 
# Results

Using exactly the same setting as in [sherjilozair's repo](https://github.com/sherjilozair/char-rnn-tensorflow), we
obtain the following results:

![Training losses](https://github.com/phvu/grid-lstm-tensorflow/raw/master/char-rnn/imgs/avg_train_losses.png "Average Training losses")

Note that the gridLSTM network was trained with exactly the same setting (learning rate, batch size etc...)
More tests with deeper networks will be added once I finish testing.