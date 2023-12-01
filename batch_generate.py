from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
import pdb
import fire
import sys
import os
import torch
import pandas as pd
from Utilities import prompter, utilities
import torch

from typing import Union

CACHE_DIR="/mnt/scratch/users/hm2066/models/huggingface/"

function_args = {
    ''
}

template_args_results = {
    'QASC_Final-Answer_QA (step-by-step)': (['question', 'answers'],
                                            ['question', 'answers', 'generated deduced', 'actual deduced', 'pred answer', 'true answer']),
    'QASC_Final-Answer_QA': (['question', 'answers'], 
                             ['question', 'answers', 'pred answer', 'true answer']),
    'QASC_Final-Answer_QAF': (['question', 'answers', 'fact_1', 'fact_2'], 
                              ['question', 'answers', 'fact 1', 'fact 2', 'pred answer', 'true answer']),
    'QASC_Final-Answer_QAF (fact 1 only)': (['question', 'answers', 'fact_1'], 
                                            ['question', 'answers', 'fact 1', 'pred answer', 'true answer']),
    'QASC_Final-Answer_QAF (fact 2 only)': (['question', 'answers', 'fact_2'], 
                                            ['question', 'answers', 'fact 2', 'pred answer', 'true answer']),
    'QASC_Final-Answer_QAFD': (['question', 'answers', 'fact_1', 'fact_2', 'deduction'], 
                               ['question', 'answers', 'fact 1', 'fact 2', 'actual deduced', 'pred answer', 'true answer']),
    'QASC-Full': (['question', 'answers', 'fact_1', 'fact_2'], 
                  ['question', 'answers', 'fact 1', 'fact 2', 'actual deduced', 'generated deduced', 'pred answer', 'true answer']),
}

default_dict = {
    "question": [],
    "answers": [],
    "fact 1": [],
    "fact 2": [],
    "generated deduced": [],
    "actual deduced": [],
    "pred answer": [],
    "true answer": [],
}

device = "cuda" if torch.cuda.is_available() else None
assert device is not None, "Cuda not available!"

def process_batch(
        input_batch, 
        result_dict,
        tokenizer,
        model, 
        prompt, 
        generation_config, 
        max_new_tokens
):
    
    inputs = tokenizer(input_batch, padding=True, return_tensors='pt').to(device)
    input_batch = []
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=max_new_tokens,
        )
    output = tokenizer.batch_decode(generation_output)
    for response in output:
        result = prompt.get_response(response)
        if result_dict.get('generated deduced') is not None:
            result_dict['generated deduced'].append(result.split("Deduce:")[-1].strip().split('\nAnswer:')[0])
        try:
            result_dict['pred answer'].append(result.split("Answer:")[-1].strip().split('\n')[0])
        except:
            result_dict['pred answer'].append(result.split("Answer:")[-1].strip())
    return input_batch

def main(
        base_model: str = "",
        load_8bit: bool = True,
        random_seed: int = 42,
        prompt_template: str = "",
        shuffle_fact1: bool = False,
        shuffle_fact2: bool = False,
        ablate_tokens_fact1: bool = False,
        ablate_tokens_fact2: bool = False,
        ablate_tokens_question: bool = False,
        num_ablations: int = 1,
        ablate_connecting_F1Q: bool = False,
        ablate_connecting_F2Q: bool = False,
        ablate_connecting_F1F2: bool = False,
        include_question_ablation: bool = False,
        ablate_matching_words_with_answers: bool = False,
        random_fact_generator: bool = False,
        frequency_ablation: Union[None, str] = None,
        output_file_name: str = "generated_response.json",
        temperature: float = 0.7,
        top_p: float = 0.75, 
        top_k: int = 40,
        num_beams: int = 4,
        max_new_tokens: int = 128,
):
    
    
    utilities.custom_set_seed(random_seed)
    set_seed(random_seed)

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='meta-llama/Llama-2-13b-chat-hf'"

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

    # Get the keys for the current template
    current_keys = template_args_results.get(prompt_template)[1]

    # Create result_dict with only the current keys
    result_dict = {key: default_dict[key] for key in current_keys}

    if ablate_connecting_F1F2 or ablate_connecting_F1Q or ablate_connecting_F2Q:
        result_dict['ablated tokens'] = []
    
    input_batch = []

    count = 0

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )

    for i in range(len(df)):
        df_entry = df.iloc[i]
        if random_fact_generator:
            df_entry['fact 1'] = " ".join(utilities.generate_random_words(len(df_entry['fact 1'].split(" "))))
            df_entry['fact 2'] = " ".join(utilities.generate_random_words(len(df_entry['fact 2'].split(" "))))

        if ablate_matching_words_with_answers:
            df_entry['fact 1'], df_entry['fact 2'] = utilities.match_answer_with_facts(df_entry['fact 1'], df_entry['fact 2'], df_entry['answers'])

        flags = [ablate_connecting_F1F2, ablate_connecting_F1Q, ablate_connecting_F2Q]
        actions = [('fact 1', 'fact 2'), ('fact 1', 'question'), ('fact 2', 'question')]
        
        for flag, (item1, item2) in zip(flags, actions):
            if flag:
                df_entry[item1], df_entry[item2 if include_question_ablation or item2 == 'fact 2' else item1] = utilities.ablate_connecting_words(
                    df_entry[item1], df_entry[item2], result_dict
                )

        # Single if-else to handle ablation and shuffling
        items = ('fact 1', 'fact 2', 'question')
        shuffles = (shuffle_fact1, shuffle_fact2, False)
        ablates = (ablate_tokens_fact1, ablate_tokens_fact2, ablate_tokens_question)
        end_chars = ('.', '.', '?')

        for item, shuffle, ablate, end_char in zip(items, shuffles, ablates, end_chars):
            if shuffle:
                df_entry[item] = utilities.string_shuffler(df_entry[item].rstrip(end_char))
            if ablate:
                df_entry[item] = utilities.random_token_ablator(df_entry[item].rstrip(end_char), num_ablations)

        # Additional data collation
        for key in result_dict:
            if 'ablated tokens' in key:
                continue
            if df_entry.get(key) is None:
                continue
            value = df_entry['deducted fact'] if key == 'actual deduced' else df_entry.get(key.replace('_', ' '))
            result_dict[key].append(value)
        
        current_args = template_args_results.get(prompt_template)[0]

        # Create a dictionary with the current arguments
        args = {arg: df_entry['deducted fact'] if arg == 'deduction' else 
                df_entry.get(arg.replace('_', ' ')) for arg in current_args}
        
        input_prompt = prompt.generate_prompt(**args)

        input_batch.append(input_prompt)
        count+=1
        if count == 3:
            input_batch = process_batch(input_batch, result_dict, tokenizer, model, prompt, generation_config, max_new_tokens)
            count = 0

    if count != 0:
        process_batch(input_batch, result_dict, tokenizer, model, prompt, generation_config, max_new_tokens)

    result_df = pd.DataFrame(result_dict)

    result_df.to_json(output_file_name, orient ='records')

        
if __name__ == "__main__":
    fire.Fire(main)
