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
from nltk.corpus import stopwords


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
    

def ablate_connecting_words(input_string1, input_string2):
    input_string1 = input_string1.translate(str.maketrans('', '', string.punctuation))
    input_string2 = input_string2.translate(str.maketrans('', '', string.punctuation))
    listed_string1 = input_string1.split(" ")
    listed_string2 = input_string2.split(" ")
    listed_string1_copy = listed_string1[:]
    listed_string2_copy = listed_string2[:]
    for token in listed_string1_copy:
        for token2 in listed_string2_copy:
            if token == token2:
                # pdb.set_trace()
                if token in listed_string1:
                    listed_string1.remove(token)
                if token2 in listed_string2:
                    listed_string2.remove(token2)

    input_string1 = " ".join(listed_string1)
    input_string2 = " ".join(listed_string2)

    return input_string1, input_string2


def random_token_ablator(input_string, num_tokens):
    listed_string = input_string.split(" ")
    for i in range(num_tokens):
        random_generated_number = random.randrange(len(listed_string))
        while listed_string[random_generated_number] in stopwords.words('english'):
            random_generated_number = random.randrange(len(listed_string))
        listed_string.pop(random_generated_number)
    input_string = " ".join(listed_string) + "."
    return input_string

def string_shuffler(input_string):
    listed_string = input_string.split(" ")
    input_string = " ".join(random.sample(listed_string, len(listed_string))) + "."
    return input_string

def main(
        base_model: str = "",
        load_8bit: bool = True,
        random_seed: int = 42,
        prompt_template: str = "",
        shuffle_fact1: bool = False,
        shuffle_fact2: bool = False,
        ablate_tokens_fact1: bool = False,
        ablate_tokens_fact2: bool = False,
        num_ablations: int = 1,
        ablate_connecting_F1Q: bool = False,
        ablate_connecting_F2Q: bool = False,
        ablate_connecting_F1F2: bool = False,
        temperature=0.7,
        top_p=0.75, 
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
):

    custom_set_seed(random_seed)
    set_seed(random_seed)

    # print(stopwords.words('english'))

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
                #    "fact 2": [],
                   "generated deduced": [],
                   "actual deduced": [],
                   "pred answer": [],
                   "true answer": [],
                   "model_full_answer": []}
    
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
        if ablate_tokens_fact1 or shuffle_fact1:
            if df_entry['fact 1'][-1] == '.':
                df_entry['fact 1'] = df_entry['fact 1'][:-1]
        if ablate_tokens_fact2 or shuffle_fact2:
            if df_entry['fact 2'][-1] == '.':
                df_entry['fact 2'] = df_entry['fact 2'][:-1]

        if ablate_connecting_F1F2:
            df_entry['fact 1'], df_entry['fact 2'] = ablate_connecting_words(df_entry['fact 1'], df_entry['fact 2'])
        elif ablate_connecting_F1Q:
            df_entry['fact 1'], _ = ablate_connecting_words(df_entry['fact 1'], df_entry['question'])
        elif ablate_connecting_F2Q:
            df_entry['fact 2'], _ = ablate_connecting_words(df_entry['fact 2'], df_entry['question'])

        if ablate_tokens_fact1:
            df_entry['fact 1'] = random_token_ablator(df_entry['fact 1'], num_ablations)
        if ablate_tokens_fact2:
            df_entry['fact 2'] = random_token_ablator(df_entry['fact 2'], num_ablations)
        
        if shuffle_fact1 and shuffle_fact2:
                df_entry['fact 1'] = string_shuffler(df_entry['fact 1'])
                df_entry['fact 2'] = string_shuffler(df_entry['fact 2'])
             
        elif shuffle_fact1:
            df_entry['fact 1'] = string_shuffler(df_entry['fact 1'])
                
        elif shuffle_fact2:
            df_entry['fact 2'] = string_shuffler(df_entry['fact 2'])
        

                     
        result_dict['question'].append(df_entry['question'])
        result_dict['answers'].append(df_entry['answers'])
        result_dict['fact 1'].append(df_entry['fact 1'])
        # result_dict['fact 2'].append(df_entry['fact 2'])
        result_dict['actual deduced'].append(df_entry['deducted fact'])
        # result_dict['generated deduced'].append(df_entry['generated deduced'])
        result_dict['true answer'].append(df_entry['answer'])
        # result_dict['true answer'].append(df_entry['true answer'])
        print(df_entry['fact 2'])
        input_prompt = prompt.generate_prompt(question=df_entry['question'],
                               answers=df_entry['answers'],
                               fact_1=df_entry['fact 1'],
                            #    fact_2=df_entry['fact 2'],
                            #    deduction=df_entry['generated deduced']
                            # deduction=df_entry['deducted fact']
                               )

        input_batch.append(input_prompt)
        count+=1
        flag+=1
        if count == 3:
            inputs = tokenizer(input_batch, padding=True, return_tensors='pt').to(device)
            input_batch = []
            count = 0
            with torch.no_grad():
                generation_output = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                )
            output = tokenizer.batch_decode(generation_output)
            # print(output)
            for response in output:
                result = prompt.get_response(response)
                result_dict['model_full_answer'].append(result)
                result_dict['generated deduced'].append(result.split("Deduce:")[-1].strip().split('\nAnswer:')[0])
                try:
                    result_dict['pred answer'].append(result.split("Answer:")[-1].strip().split('\n')[0])
                except:
                    result_dict['pred answer'].append(result.split("Answer:")[-1].strip())

    if count != 0:
            inputs = tokenizer(input_batch, padding=True, return_tensors='pt').to(device)
            count = 0
            with torch.no_grad():
                generation_output = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                )

            output = tokenizer.batch_decode(generation_output)
            for response in output:
                result = prompt.get_response(response)
                result_dict['model_full_answer'].append(result)
                result_dict['generated deduced'].append(result.split("Deduce:")[-1].strip().split('\nAnswer:')[0])
                try:
                    result_dict['pred answer'].append(result.split("Answer:")[-1].strip().split('\n')[0])
                except:
                    result_dict['pred answer'].append(result.split("Answer:")[-1].strip())

    # pdb.set_trace()
    print(len(result_dict['pred answer']))
    print(len(result_dict['question']))
    print(len(result_dict['answers']))
    print(len(result_dict['true answer']))

    result_df = pd.DataFrame(result_dict)

    result_df.to_json("generated_response_QAF (fact 1 only with full answer).json", orient ='records')

        
if __name__ == "__main__":
    fire.Fire(main)
