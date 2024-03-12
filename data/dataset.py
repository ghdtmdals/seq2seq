import json
import re

from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source = self.data[str(index)]["korean"]
        target = self.data[str(index)]["english"]

        source = list(reversed(self.preprocess_sentence(source).split()))
        ### Translation Target Starts with the <EOS> Token from the Source Sentence
        target = ["<EOS>"] + self.preprocess_sentence(target).split() + ["<EOS>"]

        # source = 

        return source, target

    def load_data(self, data_path):
        with open(data_path, "r") as f:
            all_data = json.load(f)
        
        return all_data
    
    def preprocess_sentence(self, sent):
        # 단어와 구두점 사이에 공백 추가.
        # ex) "I am a student." => "I am a student ."
        sent = re.sub(r"([?.!,¿])", r" \1", sent)

        # (알파벳, 한글, 숫자, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환.
        sent = re.sub(r"[^a-zA-Z가-힣0-9!.?]+", r" ", sent)

        # 다수의 공백을 하나의 공백으로 치환
        sent = re.sub(r"\s+", " ", sent)
        return sent
    
if __name__ == "__main__":
    dataset = TranslationDataset(data_path = "./dataset/korean_english_dataset.json")

    korean, english = dataset[0]

    breakpoint()