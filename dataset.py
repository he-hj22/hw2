import torch
from torch.utils.data import DataLoader
from utils import load_data
from torch.nn.utils.rnn import pad_sequence

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, labels, sentences, vocab):
        self.labels = torch.tensor(labels)
        sentences = [torch.tensor([vocab[word] for word in sentence]) for sentence in sentences]
        self.sentences = pad_sequence(sentences, batch_first=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.labels[idx], self.sentences[idx]
    
def get_data_loader(file_path, vocab, batch_size, shuffle=True):
    labels, sentences = load_data(file_path)
    dataset = TextDataset(labels, sentences, vocab)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

