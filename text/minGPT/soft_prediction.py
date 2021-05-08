from pathlib import Path
import glob
from random import randint
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset, DataLoader
import torch
from minGPT.mingpt.model import GPTConfig, GPT
from minGPT.mingpt.utils import set_seed

set_seed(42)

paths = glob.glob("data/raw.en/**")


class Soft_pred_dataset(Dataset):
    def __init__(self, evaluate: bool = False):
        tokenizer = ByteLevelBPETokenizer("soft_prediction-vocab.json", "soft_prediction-merges.txt")

        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        # tokenizer.enable_truncation(max_length=128)
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        src_files = [Path(file) for file in glob.glob("/home/ignace/datasets/raw.en/**")]
        for src_file in src_files:
            print("🔥", src_file)
            lines = src_file.read_text(encoding="utf-8").splitlines()
            self.examples += [x.ids for x in tokenizer.encode_batch(lines) if len(x.ids) > 128]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        start = randint(0, len(self.examples[i]) - 129)
        data = self.examples[i][start: start + 129]
        x = torch.tensor(data[:-1], dtype=torch.long)
        y = torch.tensor(data[1:], dtype=torch.long)
        return x, y


dataset = Soft_pred_dataset()
print(len(dataset))
# data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

config = GPTConfig(52_000, 128, n_head=8, n_layer=8, n_embd=512, ckpt_path="/home/ignace/torchTestModels/soft_prediction")
model = GPT(config)


# model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

from minGPT.mingpt.trainer import Trainer, TrainerConfig

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=200, batch_size=64, learning_rate=6e-4,
                      lr_decay=True)
trainer = Trainer(model, dataset, None, tconf)
trainer.train()
