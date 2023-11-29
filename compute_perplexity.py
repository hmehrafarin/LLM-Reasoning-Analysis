import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
import pdb
import fire
import sys
import os
import torch
import pandas as pd
import prompter
import numpy as np
import torch
import random
import re
from wordfreq import word_frequency
import torch.nn.functional as F

CACHE_DIR="/mnt/scratch/users/hm2066/models/huggingface/"

device = "cuda" if torch.cuda.is_available() else "cpu"

def custom_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_choices(text):
    return re.findall(r'\([A-Z]\) [^()]*', text)

def main(
        base_model: str = "",
        load_8bit: bool = True,
        random_seed: int = 42,
        prompt_template: str = "",
):

    custom_set_seed(random_seed)
    set_seed(random_seed)

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=CACHE_DIR
            )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()
    torch.compile(model)

    df = pd.read_json('eval_data.json')

    prompt = prompter.Prompter(prompt_template)
    result_dict = {"question": [],
                   "answers": [],
                   "perplexity": []}
    
    input_batch = []

    count = 0
    flag=0

    for i in range(len(df)):
        df_entry = df.iloc[i]
        result_dict['question'].append(df_entry['question'])
        result_dict['answers'].append(df_entry['answers'])
        answer_list = split_choices(df_entry['answers'])
        ppl_per_answer = {}
        for answer in answer_list:
                     

            input_prompt = prompt.generate_prompt(question=df_entry['question'],
                                   answers=df_entry['answers'],
                                   answer=answer,
                               )
            inputs = tokenizer(input_prompt, return_tensors='pt').to(device)

            with torch.no_grad():
                loss = model(inputs['input_ids'], labels=inputs['input_ids']).loss
            ppl = torch.exp(loss)
            print(ppl)
            ppl_per_answer[answer] = ppl.item()
        result_dict['perplexity'].append(ppl_per_answer)

    result_df = pd.DataFrame(result_dict)

    result_df.to_json("QA (perplexity).json", orient ='records')

        
if __name__ == "__main__":
    fire.Fire(main)
