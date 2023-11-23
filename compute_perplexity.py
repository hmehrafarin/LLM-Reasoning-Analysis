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
import string
from wordfreq import word_frequency
from typing import Union

CACHE_DIR="/mnt/scratch/users/hm2066/models/huggingface/"

if torch.cuda.is_available():
    device = "cuda"
else:
    raise ValueError("Cuda not available!")

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
        temperature=0.7,
        top_p=0.75, 
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
):

    custom_set_seed(random_seed)
    set_seed(random_seed)

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

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
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    df = pd.read_json('eval_data.json')

    prompt = prompter.Prompter(prompt_template)
    result_dict = {"question": [],
                   "answers": [],
                   "fact 1": [],
                   "fact 2": [],
                   "generated deduced": [],
                   "actual deduced": [],
                   "pred answer": [],
                   "true answer": []}
    
    input_batch = []

    count = 0
    flag=0

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    for i in range(len(df)):
        df_entry = df.iloc[i]

        answer_list = split_choices(df_entry['answers'])
        for answer in answer_list:
                     
            result_dict['question'].append(df_entry['question'])
            result_dict['answers'].append(df_entry['answers'])
            result_dict['true answer'].append(df_entry['answer'])
            input_prompt = prompt.generate_prompt(question=df_entry['question'],
                                   answers=df_entry['answers'],
                                   answer=answer,
                               )
            inputs = tokenizer(input_prompt, padding=True, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                print(torch.nn.functional.softmax(outputs.logits, dim=-1))
            
                break
    #         input_batch.append(input_prompt)
    #         count+=1
    #         flag+=1
    #         if count == 3:
    #             inputs = tokenizer(input_batch, padding=True, return_tensors='pt').to(device)
    #             input_batch = []
    #             count = 0
    #             with torch.no_grad():
    #                 generation_output = model.generate(
    #                     **inputs,
    #                     generation_config=generation_config,
    #                     max_new_tokens=max_new_tokens,
    #                 )
    #             output = tokenizer.batch_decode(generation_output)
    #             # print(output)
    #             for response in output:
    #                 # print(response)
    #                 result = prompt.get_response(response)
    #                 result_dict['generated deduced'].append(result.split("Deduce:")[-1].strip().split('\nAnswer:')[0])
    #                 try:
    #                     result_dict['pred answer'].append(result.split("Answer:")[-1].strip().split('\n')[0])
    #                 except:
    #                     result_dict['pred answer'].append(result.split("Answer:")[-1].strip())


    # if count != 0:
    #         inputs = tokenizer(input_batch, padding=True, return_tensors='pt').to(device)
    #         count = 0
    #         with torch.no_grad():
    #             generation_output = model.generate(
    #                 **inputs,
    #                 generation_config=generation_config,
    #                 max_new_tokens=max_new_tokens,
    #             )

    #         output = tokenizer.batch_decode(generation_output)
    #         for response in output:
    #             result = prompt.get_response(response)
    #             # print(result)
    #             result_dict['generated deduced'].append(result.split("Deduce:")[-1].strip().split('\nAnswer:')[0])
    #             try:
    #                 result_dict['pred answer'].append(result.split("Answer:")[-1].strip().split('\n')[0])
    #             except:
    #                 result_dict['pred answer'].append(result.split("Answer:")[-1].strip())

    # pdb.set_trace()
    # print(len(result_dict['pred answer']))
    # print(len(result_dict['question']))
    # print(len(result_dict['answers']))
    # print(len(result_dict['true answer']))

    # result_df = pd.DataFrame(result_dict)

    # result_df.to_json("generated_response-full (F1F2 connecting ablation based on frequency).json", orient ='records')

        
if __name__ == "__main__":
    fire.Fire(main)
