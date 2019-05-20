import torch
import ipdb
rnn = torch.nn.RNNCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
ipdb.set_trace()
for i in range(6):
        hx = rnn(input[i], hx)
        output.append(hx)
