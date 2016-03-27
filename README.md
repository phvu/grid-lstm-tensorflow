# grid-lstm-tensorflow

Examples of using GridLSTM (and GridRNN in general) in tensorflow

The GridRNN implementation in tensorflow is generic, in the sense that it supports multiple 
dimensions with various settings for input/output dimensions, priority dimensions and non-recurrent dimensions.
The type of recurrent cell can also be selected among LSTM, GRU or vanilla RNN.

Here we collect some examples that demonstrate GridRNN, which will be added over time.
The current list of examples include:

- `char-rnn`: 2GridLSTM for character-level language modeling.

