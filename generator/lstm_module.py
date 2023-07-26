import torch
import torch.nn as nn

class LSTMModule(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_prob):
        super(LSTMModule, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if GPU is available
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.ident = torch.eye(input_size).to(self.device)  # Make sure identity matrix is on the correct device
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True).to(self.device)  # Send module to correct device
        self.dropout = nn.Dropout(dropout_prob).to(self.device)  # Send module to correct device
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)  # Send module to correct device

    def forward(self, x):
        x = x.to(self.device)  # Make sure input tensor is on the correct device
        one_hot_list = [self.ident[t.long()] for t in x]
        one_hot = torch.stack(one_hot_list)
        one_hot = one_hot.view(x.shape[0], -1, self.input_size)
        _, (hidden, _) = self.lstm(one_hot)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return output
