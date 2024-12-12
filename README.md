#  **Reasoning or a Semblance of it? A Diagnostic Study of Transitive Reasoning in LLMs**

This is the repository for the EMNLP 2024 paper, [Reasoning or a Semblance of it? A Diagnostic Study of Transitive Reasoning in LLMs](https://aclanthology.org/2024.emnlp-main.650/)

This repository contains the code for the experiments in the paper. To run the experiments run the following lines in the terminal:
```bash
python batch_generate.py --base_model 'meta-llama/Llama-2-13b-chat-hf' \
--batch_size 2 \
--prompt_template 'Bamboogle-Full' \
--data_path 'data/updated-bamboogle-gibberish.json' \
--random_seed 123
```
This code block generates answers for the **Bamboogle Full** experiment with the **LLaMA-2** model, to add manipulations to the facts you can toggle the option correlated to that experiment to `True`. For instance:
```bash
python
python batch_generate.py --base_model 'meta-llama/Llama-2-13b-chat-hf' \
--batch_size 2 \
--prompt_template 'Bamboogle-Full' \
--data_path 'data/updated-bamboogle-gibberish.json' \
--random_seed 123 \
--shuffle_fact1 True \
--shuffle_fact2 True
```
This runs the Bamboogle both facts shuffled experiment.

