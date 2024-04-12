import pandas as pd

from datasets import Dataset


def make_dataset(fn, sample=-1):
    df = pd.read_csv(fn, sep='\t')[['document', 'label']]
    if sample > 0:
        df = df.sample(sample)
    dataset = Dataset.from_pandas(df)
    return dataset


# prompt: https://huggingface.co/google/gemma-1.1-2b-it
def gen_train_prompt(example):
    prompt_list = []
    for i in range(len(example['document'])):
        doc = example['document'][i]
        label = '긍정' if example['label'][i] == 1 else '부정'
        prompt_list.append(r"""<bos><start_of_turn>user
다음 문장은 영화리뷰입니다. 긍정 또는 부정으로 분류해주세요:

{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(doc, label))
    return prompt_list


def gen_test_prompt(example):
    prompt_list = []
    for i in range(len(example['document'])):
        doc = example['document'][i]
        prompt_list.append(r"""<bos><start_of_turn>user
다음 문장은 영화리뷰입니다. 긍정 또는 부정으로 분류해주세요:

{}<end_of_turn>
<start_of_turn>model
""".format(doc))
    return prompt_list
