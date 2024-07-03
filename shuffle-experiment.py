from src import utilities
import pandas as pd
import numpy as np

import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    )

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, 
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)

from datasets import load_dataset

import torch

device = "cuda" if torch.cuda.is_available() else None
assert device is not None, "Cuda not available!"

prompt = "Given the shuffled sentence below, please restore the original sentence:\n Context: \nShuffled sentence: of by water formed are water condensing beads vapor \nOriginal sentence: beads of water are formed by water vapor condensing\nContext:\nShuffled sentence: Clouds vapor of water made are\nOriginal sentence: Clouds are made of water vapor\nContext:\nShuffled sentence: water Condensation a to liquid change of vapor is the\n Original sentence: Condensation is the change of water vapor to a liquid\nContext:\n Shuffled sentence: "
data_path = "data/eval_data.json"
df = pd.read_json(data_path)

result_dict = {
    "shuffled_sentence": [],
    "original_sentence": [],
    "model_response": [],
    "correct": []
}

CACHE_DIR = "/mnt/scratch/users/hm2066/models/huggingface/"

end_chars = ('.', '.', '?')
list_of_items = ['fact 1', 'fact 2']
base_model = "google/flan-t5-xxl"

tokenizer = AutoTokenizer.from_pretrained(base_model)
config = transformers.AutoConfig.from_pretrained(base_model, cache_dir=CACHE_DIR) # type: ignore
AUTO_MODEL_CLASS = AutoModelForCausalLM if getattr(config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES else AutoModelForSeq2SeqLM
if AUTO_MODEL_CLASS == AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
        )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
elif AUTO_MODEL_CLASS == AutoModelForSeq2SeqLM:
    # tokenizer = AutoTokenizer.from_pretrained(base_model, model_max_length=1024)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )

count = 0
batch = []
generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        do_sample=True,
    )


dataset = load_dataset('json', data_files={'eval': 'data/eval_data.json'})

eval_data = dataset['eval']

facts = ['fact 1', 'fact 2']

for entry in facts:
    shuffled_eval_data = eval_data.map(utilities.string_shuffler, fn_kwargs={'entry': entry})

for orig_instance, shuffled_instance in zip(eval_data, shuffled_eval_data):
    for item in facts:
        if orig_instance[item].lower() in result_dict["original_sentence"]:
            continue
        result_dict["shuffled_sentence"].append(shuffled_instance[item].lower())
        result_dict["original_sentence"].append(orig_instance[item].lower())
        input_prompt = f"{prompt}{shuffled_instance[item].lower()}"
        batch.append((input_prompt, orig_instance[item].lower()))
        count += 1
        if count == 3:
            count = 0
            input_batch, target_batch = [t[0] for t in batch], [t[1] for t in batch]
            inputs = tokenizer(input_batch, padding=True, return_tensors='pt').to(device)
            model_output = model.generate(**inputs, generation_config=generation_config,max_new_tokens=128)
            batch_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)
            for output, target in zip(batch_output, target_batch):
                if AUTO_MODEL_CLASS == AutoModelForCausalLM:
                    try:
                        clean_output = output.split("Inference:\n")[1].split("Original sentence: ")[1].split("\n")[0]
                    except:
                        clean_output = ""
                else:
                    try:
                        clean_output = output.split("Original sentence: ")[1].split("\n")[0]
                    except:
                        clean_output = ""
                if clean_output.endswith('.'):
                    clean_output = clean_output[:-1]
                if target.endswith('.'):
                    target = target[:-1]
                result_dict["model_response"].append(clean_output.lower())
                result_dict["correct"].append(clean_output.lower() == target)
                print("##################")
                print(clean_output.lower())
                print(target)
                print("##################")
            batch = []

if len(batch) > 0:
    input_batch, target_batch = [t[0] for t in batch], [t[1] for t in batch]
    inputs = tokenizer(input_batch, padding=True, return_tensors='pt').to(device)
    model_output = model.generate(**inputs, generation_config=generation_config,max_new_tokens=128)
    batch_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)
    for output, target in zip(batch_output, target_batch):
        try:
            clean_output = output.split("Inference:\n")[1].split("Original sentence: ")[1].split("\n")[0]
        except:
            clean_output = ""
        if clean_output.endswith('.'):
            clean_output = clean_output[:-1]
        if target.endswith('.'):
            target = target[:-1]

        result_dict["model_response"].append(clean_output.lower())
        result_dict["correct"].append(clean_output.lower() == target)
       

result_df = pd.DataFrame(result_dict)
print(len(result_df["shuffled_sentence"]))
print(len(result_df["original_sentence"]))
print(len(result_df["model_response"]))
print(len(result_df["correct"]))

result_df.to_json("shuffled_experiment_flan.json", orient ='records')

