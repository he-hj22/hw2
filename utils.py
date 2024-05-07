import gensim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import random

def load_data(file_path):
    labels = []
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
       for line in f:
            parts = line.strip().split('\t')
            label = (int)(parts[0])
            sentence = parts[1].split()
            labels.append(label)
            sentences.append(sentence)
            
    return labels, sentences


def build_vocab():
   
    file_names = ['train', 'test', 'validation']
    vocab = {}

    for file_name in file_names:
        file_path =  f'./dataset/{file_name}.txt'
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                sentence = parts[1].split()
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = len(vocab)
    return vocab


def load_pretrained_embedding(vocab):
    file_path = 'dataset\wiki_word2vec_50.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

    embedding_matrix = np.zeros((len(vocab), 50))
    for word in vocab:
        if word in word2vec:
            embedding_matrix[vocab[word]] = word2vec[word]

    return embedding_matrix    


def draw_curves(dir, train_accuracy_list, val_accuracy_list, train_f_score_list, val_f_score_list):
    x = np.arange(1, len(train_accuracy_list) + 1)

    # 创建图表
    plt.figure(figsize=(10, 5))  # 可以调整图表大小

    # 绘制精度曲线
    plt.plot(x, train_accuracy_list, 'b-', label='Train Accuracy')
    plt.plot(x, val_accuracy_list, 'b--', label='Validation Accuracy')

    # 绘制F-score曲线
    plt.plot(x, train_f_score_list, 'r-', label='Train F-Score')
    plt.plot(x, val_f_score_list, 'r--', label='Validation F-Score')

    # 添加标题、图例和轴标签
    plt.title('Accuracy and F-Score')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(['Train Accuracy', 'Validation Accuracy', 'Train F-Score', 'Validation F-Score'])

    # 设置主要网格线
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    # 保存图表
    file_path = f'{dir}/performance.png'
    plt.savefig(file_path)
    plt.close()


def write_config(dir, args, test_acc, test_f_score):
    file_path = f'{dir}/config.txt'
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f'[Config]\n')
        for key, value in vars(args).items():
            f.write(f'- {key}={value}\n')
        f.write(f'\n[Result]\n - accuracy={100. *test_acc:.2f}% \n - f_score={100. *test_f_score:.2f}%\n')


def initiate_environment(args):
  
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)