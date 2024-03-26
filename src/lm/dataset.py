import collections

import torch


class LanguageModelDataset(torch.utils.data.Dataset):

    def __init__(self, fn, bos="<s>", eos="</s>"):
        super().__init__()

        self.idx2label = collections.OrderedDict()  # not used
        self.label2idx = collections.OrderedDict()  # not used
        self.fn = fn
        self.bos = bos
        self.eos = eos

        lines = []
        with open(fn, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        self.texts = lines

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return f"{self.bos}{self.texts[idx]}", f"{self.texts[idx]}{self.eos}"


class LanguageModelCollator:

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        dec_inputs, dec_labels = zip(*batch)

        dec_inputs = self.tokenizer(
            dec_inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dec_labels = self.tokenizer(
            dec_labels,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dec_labels = dec_labels["input_ids"]
        dec_labels[dec_labels == self.tokenizer.pad_token_id] = -100

        return_value = {
            "input_ids": dec_inputs["input_ids"],
            "attention_mask": dec_inputs["attention_mask"],
            "labels": dec_labels,
        }

        return return_value
