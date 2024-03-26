import os
import argparse

from tokenizers import ByteLevelBPETokenizer  # CharBPETokenizer


def define_config():
    p = argparse.ArgumentParser(description="Train tokenizer")

    p.add_argument("--train_files", type=str, nargs="+", required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--min_frequency", type=int, default=2)

    config = p.parse_args()

    return config


def main():
    config = define_config()

    # reserved 100 tokens
    unused_tokens = [f"<unused_{i}>" for i in range(100)]

    # train tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=config.train_files,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=[
            "<pad>",  # padding
            "<s>",  # start of sentence
            "</s>",  # end of sentence
            "<unk>",  # unknown
        ]
        + unused_tokens,
        show_progress=True,
    )
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    # save tokenizer
    os.makedirs(config.output_dir, exist_ok=True)
    tokenizer.save(os.path.join(config.output_dir, "tokenizer.json"))

    ko_sentence = '<s>이것은 테스트 문장입니다. <unused_0>어떻게 보이나요?<unused_1> 고유명사 "파이썬 파이토치 허깅페이스"는 어떻게 되나요?</s>'
    en_sentence = '<s>This is a test sentence. <unused_0>How does it look?<unused_1> Proper nouns "Python PyTorch HuggingFace" how does it go?</s>'

    print(ko_sentence)
    print(">>>", tokenizer.encode(ko_sentence).tokens)
    print(">>>", tokenizer.decode(tokenizer.encode(ko_sentence).ids))
    print(en_sentence)
    print(">>>", tokenizer.encode(en_sentence).tokens)
    print(">>>", tokenizer.decode(tokenizer.encode(en_sentence).ids))


if __name__ == "__main__":
    main()
