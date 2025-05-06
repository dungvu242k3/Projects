import numpy as np
import torch
import torch.nn as nn

n_features = 2
hidden_dim = 2

torch.manual_seed(19)
rnn_cell = nn.RNNCell(input_size = n_features,hidden_size = hidden_dim)
rnn_state = rnn_cell.state_dict()
print(rnn_state)

initial_hidden = torch.zeros(1,hidden_dim)
linear_hidden = nn.linear(hidden_dim,hidden_dim)
with torch.no_grad():
    linear_hidden.weight = nn.Parameter(rnn_state['weight_hh'])
    linear_hidden.bias = nn.Parameter(rnn_state['bias_hh'])
th = linear_hidden(initial_hidden)
print(th)

linear_input = nn.Linear(n_features,hidden_dim)
with torch.no_grad():
    linear_input.weight = nn.Parameter(rnn_state['weight_ih'])
    linear_input.bias = nn.Parameter(rnn_state['bias_ih'])

X = torch.as_tensor(points[0]).float()  # Dữ liệu từ dataset
tx = linear_input(X[0:1])
print(tx)

adding = th+tx
print(adding)

hidden_updated = torch.tanh(adding)
print(hidden_updated)


