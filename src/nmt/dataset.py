import collections
from tqdm import tqdm

from torch.utils.data import Dataset


class TranslationDatatset(Dataset):

    def __init__(self, src_fn, tgt_fn, bos="<s>", eos="</s>"):
        self.idx2label = collections.OrderedDict()  # not used
        self.label2idx = collections.OrderedDict()  # not used
        self.src_fn = src_fn
        self.tgt_fn = tgt_fn
        self.bos = bos
        self.eos = eos

        self.src_data = []
        self.tgt_data = []

        with open(src_fn, "r") as f:
            for line in tqdm(f, desc="Loading source data"):
                self.src_data.append(line.strip())

        with open(tgt_fn, "r") as f:
            for line in tqdm(f, desc="Loading target data"):
                self.tgt_data.append(line.strip())

        assert len(self.src_data) == len(self.tgt_data)

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return (
            self.src_data[idx],
            f"{self.bos}{self.tgt_data[idx]}",
            f"{self.tgt_data[idx]}{self.eos}",
        )


class TranslationCollator:

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        enc_batch, dec_batch, tgt_batch = zip(*batch)

        encoder_input = self.tokenizer(
            enc_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        decoder_input = self.tokenizer(
            dec_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        decoder_labels = self.tokenizer(
            tgt_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        decoder_labels["input_ids"][
            decoder_labels["input_ids"] == self.tokenizer.pad_token_id
        ] = -100

        return {
            "input_ids": encoder_input["input_ids"],
            "attention_mask": encoder_input["attention_mask"],
            "decoder_input_ids": decoder_input["input_ids"],
            "labels": decoder_labels["input_ids"],
        }
