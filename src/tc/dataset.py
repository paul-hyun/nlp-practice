import collections

import torch


class TextClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, filename):
        super().__init__()

        self.filename = filename

        with open(filename, "r") as f:
            lines = [
                line.split("\t") for line in f.readlines() if len(line.split("\t")) == 2
            ]

        self.labels = [line[0].strip() for line in lines]
        self.texts = [line[1].strip() for line in lines]

        self.label2idx = collections.OrderedDict()
        self.idx2label = collections.OrderedDict()

        for idx, label in enumerate(set(self.labels)):
            self.label2idx[label] = idx
            self.idx2label[idx] = label

        self.n_classes = len(self.label2idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.label2idx[self.labels[idx]]


class TextClassificationCollator:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        texts, labels = zip(*samples)

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return_value = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        return return_value
