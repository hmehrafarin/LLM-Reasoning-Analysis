# Reasoning or a Semblance of it?

[![ci](https://github.com/hmehrafarin/reasoning-or-semblance/actions/workflows/ci.yml/badge.svg)](https://github.com/hmehrafarin/reasoning-or-semblance/actions/workflows/ci.yml)

Code and data for **Reasoning or a Semblance of it? A Diagnostic Study of Transitive Reasoning in LLMs**
(Houman Mehrafarin, Arash Eshghi, Ioannis Konstas; EMNLP 2024).
[Paper](https://aclanthology.org/2024.emnlp-main.650/)

The paper asks whether LLMs answering two-hop questions actually chain the two supporting facts
(A → B, B → C ⇒ A → C) or lean on shortcuts. It runs LLaMA 2 and Flan-T5 on QASC and Bamboogle
under controlled manipulations of the test query: shuffling the words in the facts, removing the
words a fact shares with the question or the other fact, removing answer keywords from the facts,
and replacing named entities in the answers with gibberish.

## Install

```bash
git clone https://github.com/hmehrafarin/reasoning-or-semblance
cd reasoning-or-semblance
uv sync --extra cpu        # laptop or CI; use --extra gpu on a CUDA machine
```

Without uv: `pip install -e ".[cpu]"` or `pip install -e ".[gpu]"`.

## Run an experiment

```bash
transitive-reasoning list                                              # every experiment and the table it reproduces
transitive-reasoning run --experiment qasc/full --model llama2_13b_chat # generate and score
transitive-reasoning evaluate results/qasc/full/llama2_13b_chat/seed42/predictions.jsonl
transitive-reasoning restore-word-order --model flan_t5_xxl            # Section 5.1
```

Each run writes `predictions.jsonl` (one row per instance with the manipulated inputs, the
generation and what was parsed from it), `metrics.json` and `run.json` (configs, seed, git
commit and library versions) under `results/<experiment>/<model>/seed<seed>/`.
Add `--limit 5` to smoke-test on a few instances; `--seed` defaults to 42.

The paper's models are 8-bit quantised and need a CUDA GPU (`configs/models/`). Any Hugging Face
causal or seq2seq model works: copy a model config, change `hf_id`, and set `quantization: none`
to run on CPU.

## Experiments

| Experiment | Dataset | Prompt | Manipulations | Metric | Paper |
|---|---|---|---|---|---|
| `bamboogle/f1f2_ablation` | bamboogle | `bamboogle/full` | remove_shared_words(fact1, fact2) on fact1, fact2 | rouge1 | Table 3 |
| `bamboogle/f1q_ablation` | bamboogle | `bamboogle/full` | remove_shared_words(fact1, question) on fact1 | rouge1 | Table 3 |
| `bamboogle/f2q_ablation` | bamboogle | `bamboogle/full` | remove_shared_words(fact2, question) on fact2 | rouge1 | Table 3 |
| `bamboogle/full` | bamboogle | `bamboogle/full` | none | rouge1 | Table 3 |
| `bamboogle/full_shuffled` | bamboogle | `bamboogle/full` | shuffle_words(fact1, fact2) | rouge1 | Table 3 |
| `bamboogle/qa` | bamboogle | `bamboogle/qa` | none | rouge1 | Table 3 |
| `bamboogle/qa_step_by_step` | bamboogle | `bamboogle/qa_step_by_step` | none | rouge1 | Table 3 |
| `bamboogle/qaf` | bamboogle | `bamboogle/qaf` | none | rouge1 | Table 3 |
| `bamboogle/qaf_fact1_only` | bamboogle | `bamboogle/qaf_fact1_only` | none | rouge1 | Table 3 |
| `bamboogle/qaf_fact2_only` | bamboogle | `bamboogle/qaf_fact2_only` | none | rouge1 | Table 3 |
| `bamboogle_gibberish/full` | bamboogle_gibberish | `bamboogle/full` | none | rouge1 | Table 4 |
| `bamboogle_gibberish/full_shuffled` | bamboogle_gibberish | `bamboogle/full` | shuffle_words(fact1, fact2) | rouge1 | Table 4 |
| `qasc/f1f2_ablation` | qasc | `qasc/full` | remove_shared_words(fact1, fact2) on fact1, fact2 | mc_accuracy | Table 2 |
| `qasc/f1f2a_keyword_ablation` | qasc | `qasc/full` | remove_answer_keywords(fact1, fact2) | mc_accuracy | Table 2 |
| `qasc/f1q_ablation` | qasc | `qasc/full` | remove_shared_words(fact1, question) on fact1 | mc_accuracy | Table 2 |
| `qasc/f2q_ablation` | qasc | `qasc/full` | remove_shared_words(fact2, question) on fact2 | mc_accuracy | Table 2 |
| `qasc/full` | qasc | `qasc/full` | none | mc_accuracy | Table 1 |
| `qasc/full_shuffled` | qasc | `qasc/full` | shuffle_words(fact1, fact2) | mc_accuracy | Figure 2 |
| `qasc/qa` | qasc | `qasc/qa` | none | mc_accuracy | Table 1 |
| `qasc/qa_step_by_step` | qasc | `qasc/qa_step_by_step` | none | mc_accuracy | Table 1 |
| `qasc/qaf` | qasc | `qasc/qaf` | none | mc_accuracy | Table 1 |
| `qasc/qaf_fact1_only` | qasc | `qasc/qaf_fact1_only` | none | mc_accuracy | Table 1 |
| `qasc/qaf_fact2_only` | qasc | `qasc/qaf_fact2_only` | none | mc_accuracy | Table 1 |

## Layout

```
configs/experiments/   one YAML per experiment in the paper
configs/models/        LLaMA 2 13b and 7b chat, Flan-T5 XXL, with the paper's decoding settings
prompts/               instruction plus three demonstrations per prompt variant (Appendix C)
data/                  QASC dev split and the re-annotated Bamboogle sets (see data/README.md)
src/transitive_reasoning/
  data.py              Instance records and dataset loading
  prompts.py           prompt rendering and response parsing
  manipulations.py     shuffle_words, remove_shared_words, remove_answer_keywords
  models.py            Hugging Face backend
  run.py               the experiment pipeline
  evaluate.py          mc_accuracy, rouge1, exact_match
  restore_word_order.py
  cli.py
tests/
```

## Development

```bash
uv sync --extra cpu
uv run pre-commit install
uv run pytest                 # unit tests, CPU only
uv run pytest -m integration  # downloads two tiny models
uv run ruff check . && uv run mypy
```

## Citation

```bibtex
@inproceedings{mehrafarin-etal-2024-reasoning,
  title     = {Reasoning or a Semblance of it? A Diagnostic Study of Transitive Reasoning in {LLM}s},
  author    = {Mehrafarin, Houman and Eshghi, Arash and Konstas, Ioannis},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year      = {2024},
  pages     = {11647--11662},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.emnlp-main.650/}
}
```

## Licence

Code is MIT licensed. Data licences are listed in `data/README.md`.
