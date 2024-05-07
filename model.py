import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters,  pretrained_embeddings, num_class=2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        weights = torch.from_numpy(pretrained_embeddings).float()
        self.embedding.weight = nn.Parameter(weights, requires_grad=True)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (kernel_size, embedding_dim))
            for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_class)

        torch.nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1) 
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in x]  
        x = torch.cat(x, 1)
        return self.fc(x)

    
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        weights = torch.from_numpy(pretrained_embeddings).float()
        self.embedding.weight = nn.Parameter(weights, requires_grad=True)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc4 = nn.Linear(hidden_dim // 4, output_dim)
  

        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)
        torch.nn.init.kaiming_uniform_(self.fc4.weight)

    def forward(self, x):
        x = self.embedding(x) 
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class RNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(RNN_LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        weights = torch.from_numpy(pretrained_embeddings).float()
        self.embedding.weight = nn.Parameter(weights, requires_grad=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        torch.nn.init.kaiming_normal_(self.lstm.weight_ih_l0)
        torch.nn.init.kaiming_normal_(self.lstm.weight_hh_l0)
        torch.nn.init.kaiming_normal_(self.lstm.weight_ih_l0_reverse)
        torch.nn.init.kaiming_normal_(self.lstm.weight_hh_l0_reverse)

        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.embedding(x) 
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RNN_GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(RNN_GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        weights = torch.from_numpy(pretrained_embeddings).float()        
        self.embedding.weight = nn.Parameter(weights, requires_grad=True)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        torch.nn.init.kaiming_normal_(self.gru.weight_ih_l0)
        torch.nn.init.kaiming_normal_(self.gru.weight_hh_l0)
        torch.nn.init.kaiming_normal_(self.gru.weight_ih_l0_reverse)
        torch.nn.init.kaiming_normal_(self.gru.weight_hh_l0_reverse)

        torch.nn.init.kaiming_uniform_(self.fc1.weight)
        torch.nn.init.kaiming_uniform_(self.fc2.weight)
        torch.nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self, x):
        x = self.embedding(x) 
        x, _ = self.gru(x) 
        x = x.mean(dim=1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)