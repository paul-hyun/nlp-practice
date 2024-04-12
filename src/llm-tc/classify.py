import argparse

from tqdm.auto import tqdm

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

from data import (
    make_dataset,
    gen_test_prompt
)

MODEL_ID = "google/gemma-1.1-2b-it"


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)
    p.add_argument("--test_tsv_fn", required=True)

    config = p.parse_args()

    return config


def main(config):
    print(config)
    
    # bits and bytes config (4bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(config.model_fn,
                                                 device_map="auto",
                                                 quantization_config=bnb_config)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              add_special_tokens=True)
    tokenizer.padding_side = 'right'

    # test dataset
    test_dataset = make_dataset(config.test_tsv_fn, sample=10)
    # pipeline
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=5)
    
    # infer
    for example in test_dataset.iter(1):
        doc = example['document'][0]

        prompt = gen_test_prompt(example)
        outputs = pipe(
            prompt,
            do_sample=True,
            temperature=0.2,
            top_k=50,
            top_p=0.95,
            add_special_tokens=True
        )
        pred = outputs[0][0]['generated_text'][len(prompt[0]):]
        print(f"- {doc} : {pred}")


if __name__ == "__main__":
    config = define_config()
    main(config)
