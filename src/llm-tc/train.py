import os
import argparse
from datetime import datetime

from tqdm.auto import tqdm
import wandb

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

from data import (
    make_dataset,
    gen_train_prompt,
    gen_test_prompt
)


MODEL_ID = "google/gemma-1.1-2b-it"


def define_config():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", required=True)
    p.add_argument("--train_tsv_fn", required=True)
    p.add_argument("--test_tsv_fn", type=str, default=None)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=0.03)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    p.add_argument("--logging_steps", type=int, default=100)

    p.add_argument("--skip_wandb", action="store_true")

    config = p.parse_args()

    return config


def get_now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def wandb_init(config):
    final_model_name = f"{config.model_name}-{get_now()}"
    
    if config.skip_wandb:
        return final_model_name

    wandb.login()
    wandb.init(
        project="NLP_EXP_rnn_text_classification",
        config=vars(config),
        id=final_model_name,
    )
    wandb.run.name = final_model_name
    wandb.run.save()

    os.makedirs(config.output_dir, exist_ok=True)

    return final_model_name


def main(config):
    print(config)

    # train dataset
    train_dataset = make_dataset(config.train_tsv_fn)

    # lora config
    lora_config = LoraConfig(
        r=6,
        lora_alpha = 8,
        lora_dropout = 0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # bits and bytes config (4bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 device_map="auto",
                                                 quantization_config=bnb_config)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              add_special_tokens=True)
    tokenizer.padding_side = 'right'

    final_model_name = wandb_init(config)

    # trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        max_seq_length=config.max_seq_length,
        args=TrainingArguments(
            output_dir=os.path.join(config.output_dir, final_model_name),
            # num_train_epochs = 1,
            max_steps=config.max_steps,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim="paged_adamw_8bit",
            warmup_steps=config.warmup_steps,
            learning_rate=config.lr,
            bf16=True,
            # fp16=True,
            logging_steps=config.logging_steps,
            push_to_hub=False,
            report_to="wandb" if not config.skip_wandb else None,
        ),
        peft_config=lora_config,
        formatting_func=gen_train_prompt,
    )
    # train
    trainer.train()
    # save lora (delta weight)
    trainer.model.save_pretrained("lora_adapter")
    # original model load (before finetuned)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 device_map='auto',
                                                 torch_dtype=torch.float16)
    # merge : original + delta wieght
    model = PeftModel.from_pretrained(model,
                                      "lora_adapter",
                                      device_map='auto',
                                      torch_dtype=torch.float16)
    model = model.merge_and_unload()
    # save fine-tunned model
    model.save_pretrained(os.path.join(config.output_dir, final_model_name, "checkpoint-final"))

    if not config.test_tsv_fn is None:
        # test dataset
        test_dataset = make_dataset(config.test_tsv_fn)
        # pipeline
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=5)
        
        # infer
        total_sample_cnt, total_correct_cnt = 0, 0
        for example in tqdm(test_dataset.iter(1)):
            label = '긍정' if example['label'][0] == 1 else '부정'

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
            total_sample_cnt += 1
            total_correct_cnt += 1 if label == pred else 0
            if total_sample_cnt >= 1000:
                break
        print(f"Test Accuracy: {total_correct_cnt} / {total_sample_cnt} = {total_correct_cnt/total_sample_cnt:.4f}")
        if not config.skip_wandb:
            wandb.log(
                {
                    "test/accuracy": total_correct_cnt / total_sample_cnt * 100,
                }
            )
            
    if not config.skip_wandb:
        wandb.finish()


if __name__ == "__main__":
    config = define_config()
    main(config)
