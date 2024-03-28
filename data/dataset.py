import os
import json
import torch
from torch.utils.data import Dataset

### Data Source: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=126

class TranslationDataset(Dataset):
    @staticmethod
    def load_vocab():
        if os.path.isfile("./dataset/korean_vocab.json") and os.path.isfile("./dataset/english_vocab.json"):
            with open("./dataset/korean_vocab.json", "r") as f:
                source_vocab = json.load(f)
            
            with open("./dataset/english_vocab.json", "r") as f:
                target_vocab = json.load(f)

        else:
            print("No Vocabulary Found")
        
        print("Korean Vocabulary Size: {:d} | English Vocabulary Size: {:d}".format(len(source_vocab), len(target_vocab)))
        
        return source_vocab, target_vocab
    
    source_vocab, target_vocab = load_vocab()

    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_indices = list(self.data.keys())

        ### Source Sentence is Reversed
        source_sent = list(reversed(self.data[data_indices[index]]["korean"].split("/")))

        ### Target Sequence's First Input Token is <EOS> Token from Source Sentence
        target_sent = ["<EOS>"] + self.data[data_indices[index]]["english"].split("/")  + ["<EOS>"]

        source_sent, target_sent = self.convert_to_sequence(source_sent, target_sent)

        return source_sent, target_sent
    
    def convert_to_sequence(self, source, target):
        source_seq = []
        for token in source:
            source_seq.append(TranslationDataset.source_vocab[token])
        
        target_seq = []
        for token in target:
            target_seq.append(TranslationDataset.target_vocab[token])
        
        source_seq = torch.Tensor(source_seq).type(torch.int32)
        target_seq = torch.Tensor(target_seq).type(torch.int32)

        return source_seq, target_seq

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            all_data = json.load(f)
        
        return all_data
    
if __name__ == "__main__":
    # source_vocab, target_vocab = TranslationDataset.load_vocab()
    dataset = TranslationDataset(data_path = "./dataset/train_korean_english_dataset.json")

    korean, english = dataset[0]

    breakpoint()