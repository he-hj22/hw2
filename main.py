import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loader
from model import TextCNN, MLP, RNN_LSTM, RNN_GRU
from utils import build_vocab, load_pretrained_embedding, draw_curves, write_config, initiate_environment
from tqdm import tqdm
import os

def train_one_epoch(model, optimizer, train_loader, device, accuracy_list, f_socre_list):
    print(f'[mode: train]')
    
    model.train()
    for batch_idx, (target, data) in tqdm(enumerate(train_loader), total=len(train_loader)):
        
        correct, total = 0, 0
        correct_label0, total_label0 = 0, 0
        correct_label1, total_label1 = 0, 0

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        correct += (output.argmax(1) == target).float().sum().item()
        total += len(target)
        correct_label0 += ((output.argmax(1) == target) & (target == 0)).float().sum().item()
        total_label0 += (target == 0).float().sum().item()
        correct_label1 += ((output.argmax(1) == target) & (target == 1)).float().sum().item()
        total_label1 += (target == 1).float().sum().item()
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            tqdm.write(f'[Batch: {batch_idx+1}] \tLoss: {loss.item():.4f}')

    if correct_label1 == 0:
        precision = 0
    else:    
        precision = correct_label1 / (correct_label1 + total_label0 - correct_label0)
    recall = correct_label1 / total_label1
    accuracy = correct / total
    f_score = 2 * precision * recall / (precision + recall + 1e-8)

    accuracy_list.append(accuracy)
    f_socre_list.append(f_score)

def evaluate(mode, model, eval_loader, device, accuracy_list = None, f_score_list = None):
    print(f'[mode: {mode}]')
    model.eval()

    correct, total = 0, 0
    correct_label0, total_label0 = 0, 0
    correct_label1, total_label1 = 0, 0
    with torch.no_grad():
        for _, (target, data) in enumerate(eval_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            correct += (output.argmax(1) == target).float().sum().item()
            total += len(target)

            correct_label0 += ((output.argmax(1) == target) & (target == 0)).float().sum().item()
            total_label0 += (target == 0).float().sum().item()

            correct_label1 += ((output.argmax(1) == target) & (target == 1)).float().sum().item()
            total_label1 += (target == 1).float().sum().item()

    if correct_label1 == 0:
        precision = 0
    else:    
        precision = correct_label1 / (correct_label1 + total_label0 - correct_label0)
    recall = correct_label1 /  total_label1
    accuracy = correct / total
    f_score = 2 * precision * recall / (precision + recall+ 1e-8)
    print(f'[accuracy]: {100. *accuracy:.2f}% && [f-score]: {100. * f_score:.2f}%')
    if accuracy_list is not None and f_score_list is not None:
        accuracy_list.append(accuracy)
        f_score_list.append(f_score)
    return accuracy, f_score    
   
train_accuracy_list, train_f_socre_list = [], []
val_accuracy_list, val_f_socre_list = [], []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP',choices=['TextCNN', 'MLP', 'LSTM', 'GRU'] ,help='model to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for layer')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    args = parser.parse_args()
    initiate_environment(args)
    
    vocab = build_vocab()
    vocab_size = len(vocab) 
    pretrained_embedding = load_pretrained_embedding(vocab)
    os.makedirs('./model_output', exist_ok=True)    

    if args.model == 'TextCNN':
        model = TextCNN(vocab_size=vocab_size, embedding_dim=50, num_filters=10, kernel_sizes=[3,5,7], pretrained_embeddings=pretrained_embedding, num_class=2)
    elif args.model == 'MLP':
        model = MLP(vocab_size=vocab_size, embedding_dim=50, hidden_dim=args.hidden_dim, output_dim=2, pretrained_embeddings=pretrained_embedding)
    elif args.model == 'LSTM':
        model = RNN_LSTM(vocab_size=vocab_size, embedding_dim=50, hidden_dim=args.hidden_dim, output_dim=2, pretrained_embeddings=pretrained_embedding)
    elif args.model == 'GRU':
        model = RNN_GRU(vocab_size=vocab_size, embedding_dim=50, hidden_dim=args.hidden_dim, output_dim=2, pretrained_embeddings=pretrained_embedding)
    model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    directory = f'./model_output/{args.model}'
    os.makedirs(directory, exist_ok=True)
   

    train_loader = get_data_loader(file_path='./dataset/train.txt', vocab=vocab, batch_size=args.batch_size ,shuffle=True)
    val_loader = get_data_loader(file_path='./dataset/validation.txt',vocab=vocab, batch_size=args.batch_size, shuffle=False)
    test_loader = get_data_loader(file_path='./dataset/test.txt', vocab=vocab, batch_size=args.batch_size, shuffle=False)

    for epoch in range(args.epochs):
        print(f'======>  [Epoch {epoch+1}/{args.epochs}]')
        train_one_epoch(model, optimizer, train_loader, args.device, train_accuracy_list, train_f_socre_list)
        evaluate('validation' ,model, val_loader, args.device, val_accuracy_list, val_f_socre_list)
    test_accuracy, test_f_score =evaluate('test' ,model, test_loader, args.device)

    draw_curves(directory, train_accuracy_list, val_accuracy_list, train_f_socre_list, val_f_socre_list)
    write_config(directory, args, test_accuracy, test_f_score)