import torch
from torch import nn

from models.seq2seq import Seq2Seq

from data.dataset import TranslationDataset
from torch.utils.data import DataLoader

from torch.optim import AdamW
from tqdm import tqdm

class Train:
    def __init__(self, train_path, test_path, batch_size, epochs, learning_rate, n_workers = 0):
        self.train_path = train_path
        self.test_path = test_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_workers = n_workers
    
    def load_dataset(self, data_path):
        
        def pad_batch(batch):
            source_len = max([len(sentence[0]) for sentence in batch])
            target_len = max([len(sentence[1]) for sentence in batch])

            source_seq = torch.zeros(len(batch), source_len, dtype = torch.int32)
            target_input_seq = torch.zeros(len(batch), target_len, dtype = torch.int32)
            target_output_seq = torch.zeros(len(batch), target_len, dtype = torch.int32)
            for i, data in enumerate(batch):
                source_seq[i, :len(data[0])] = data[0]
                target_input_seq[i, :len(data[1]) - 1] = data[1][:-1]
                target_output_seq[i, :len(data[1]) - 1] = data[1][1:]

            return source_seq, target_input_seq, target_output_seq

        dataset = TranslationDataset(data_path)

        dataloader = DataLoader(dataset = dataset, batch_size = self.batch_size, shuffle = True, num_workers = self.n_workers, collate_fn = pad_batch)

        return dataloader

    def train_setup(self):
        model = Seq2Seq(n_layers = 4, input_dim = 256, hidden_dim = 512)

        criterion = nn.CrossEntropyLoss(ignore_index = 0, reduction = "sum") ### 0은 <PAD> 토큰이기 때문에 무시
        optimizer = AdamW(params = model.parameters(), lr = self.learning_rate)

        return model, criterion, optimizer
    
    def train_loop(self):
        train_dataloader = self.load_dataset(self.train_path)
        test_dataloader = self.load_dataset(self.test_path)

        model, criterion, optimizer = self.train_setup()
        model = model.to(self.device)

        model.train()
        running_loss = 0
        with torch.autograd.detect_anomaly():
            for korean, english, eng_target in tqdm(train_dataloader, desc = f"Loss: {running_loss}", leave = True):
                korean = korean.to(self.device)
                english = english.to(self.device)

                outputs = model(korean, english)

                outputs = outputs.view(-1, outputs.shape[-1])
                eng_target = eng_target.view(-1).type(torch.LongTensor).to(self.device)
                
                loss = criterion(outputs, eng_target)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

if __name__ == "__main__":
    train_path = "./dataset/train_korean_english_dataset.json"
    test_path = "./dataset/test_korean_english_dataset.json"

    batch_size = 4
    learning_rate = 1e-4
    epochs = 30

    train = Train(train_path, test_path, batch_size, epochs, learning_rate)
    train.train_loop()

    breakpoint()