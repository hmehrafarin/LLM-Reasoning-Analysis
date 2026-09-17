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

The package reads `data/`, `prompts/` and `configs/` from the repository root, so install it from
a clone (or point `TRANSITIVE_REASONING_ROOT` at one). On Linux, `pip install -e ".[cpu]"` installs
PyPI's default CUDA build of torch; use uv, or install torch from
https://download.pytorch.org/whl/cpu first, for a CPU-only wheel.

## Run an experiment

```bash
uv run transitive-reasoning list                                              # every experiment and the table it reproduces
uv run transitive-reasoning run --experiment qasc/full --model llama2_13b_chat # generate and score
uv run transitive-reasoning evaluate results/qasc/full/llama2_13b_chat/seed42/predictions.jsonl
uv run transitive-reasoning restore-word-order --model flan_t5_xxl            # Section 5.1
uv run transitive-reasoning prepare-qasc QASC_Dataset/dev.jsonl   # rebuild data/qasc/dev.json from the official release
```

If you activate the environment (`source .venv/bin/activate`) you can drop the `uv run` prefix.

Each run writes `predictions.jsonl` (one row per instance with the manipulated inputs, the
generation and what was parsed from it, including a per-instance `score`), `metrics.json` and
`run.json` (configs, seed, git commit and library versions) under
`results/<experiment>/<model>/seed<seed>/`. Add `--save-prompts` to also store the full prompt on
every row, and `--limit 5` to smoke-test on a few instances; `--seed` defaults to 42.
`transitive-reasoning list --markdown` prints the experiment table below.

The paper's models are 8-bit quantised and need a CUDA GPU (`configs/models/`). Any Hugging Face
causal or seq2seq model works: copy a model config, change `hf_id`, and set `quantization: none`
to run on CPU.

## Experiments

| Experiment | Dataset | Prompt | Manipulations | Metric | Paper |
|---|---|---|---|---|---|
| `bamboogle/f1f2_ablation` | bamboogle | `bamboogle/full` | remove_shared_words(fact1, fact2) on fact1, fact2 | rouge1 | Table 3, F1F2 Connecting Words Ablation |
| `bamboogle/f1q_ablation` | bamboogle | `bamboogle/full` | remove_shared_words(fact1, question) on fact1 | rouge1 | Table 3, F1Q Connecting Words Ablation |
| `bamboogle/f2q_ablation` | bamboogle | `bamboogle/full` | remove_shared_words(fact2, question) on fact2 | rouge1 | Table 3, F2Q Connecting Words Ablation |
| `bamboogle/full` | bamboogle | `bamboogle/full` | none | rouge1 | Table 3, Full |
| `bamboogle/full_shuffled` | bamboogle | `bamboogle/full` | shuffle_words(fact1, fact2) | rouge1 | Table 3, Full (both facts shuffled) |
| `bamboogle/qa` | bamboogle | `bamboogle/qa` | none | rouge1 | Table 3, QA |
| `bamboogle/qa_step_by_step` | bamboogle | `bamboogle/qa_step_by_step` | none | rouge1 | Table 3, QA (step-by-step) |
| `bamboogle/qaf` | bamboogle | `bamboogle/qaf` | none | rouge1 | Table 3, QAF |
| `bamboogle/qaf_fact1_only` | bamboogle | `bamboogle/qaf_fact1_only` | none | rouge1 | Table 3, QAF (fact 1 only) |
| `bamboogle/qaf_fact2_only` | bamboogle | `bamboogle/qaf_fact2_only` | none | rouge1 | Table 3, QAF (fact 2 only) |
| `bamboogle_gibberish/full` | bamboogle_gibberish | `bamboogle/full` | none | rouge1 | Table 4, Gibberish Full |
| `bamboogle_gibberish/full_shuffled` | bamboogle_gibberish | `bamboogle/full` | shuffle_words(fact1, fact2) | rouge1 | Table 4, Gibberish Both Facts Shuffled |
| `qasc/f1f2_ablation` | qasc | `qasc/full` | remove_shared_words(fact1, fact2) on fact1, fact2 | mc_accuracy | Table 2, F1F2 Connecting Words Ablation |
| `qasc/f1f2a_keyword_ablation` | qasc | `qasc/full` | remove_answer_keywords(fact1, fact2) | mc_accuracy | Table 2, F1F2A Keyword Ablation |
| `qasc/f1q_ablation` | qasc | `qasc/full` | remove_shared_words(fact1, question) on fact1 | mc_accuracy | Table 2, F1Q Connecting Words Ablation |
| `qasc/f2q_ablation` | qasc | `qasc/full` | remove_shared_words(fact2, question) on fact2 | mc_accuracy | Table 2, F2Q Connecting Words Ablation |
| `qasc/full` | qasc | `qasc/full` | none | mc_accuracy | Table 1, Full |
| `qasc/full_shuffled` | qasc | `qasc/full` | shuffle_words(fact1, fact2) | mc_accuracy | Figure 2, both facts shuffled |
| `qasc/qa` | qasc | `qasc/qa` | none | mc_accuracy | Table 1, QA |
| `qasc/qa_step_by_step` | qasc | `qasc/qa_step_by_step` | none | mc_accuracy | Table 1, QA (step-by-step) |
| `qasc/qaf` | qasc | `qasc/qaf` | none | mc_accuracy | Table 1, QAF |
| `qasc/qaf_fact1_only` | qasc | `qasc/qaf_fact1_only` | none | mc_accuracy | Table 1, QAF (fact 1 only) |
| `qasc/qaf_fact2_only` | qasc | `qasc/qaf_fact2_only` | none | mc_accuracy | Table 1, QAF (fact 2 only) |

## Layout

```
configs/experiments/   one YAML per experiment in the paper
configs/models/        LLaMA 2 13b and 7b chat, Flan-T5 XXL, with the paper's decoding settings
prompts/               instruction plus three demonstrations per prompt variant (Appendix C), byte-for-byte as used for the paper, including a literal "\n" typo in two QASC preambles that is kept on purpose
data/                  QASC dev split and the re-annotated Bamboogle sets (see data/README.md)
src/transitive_reasoning/
  paths.py             repository-root discovery for data, prompts and configs
  data.py              Instance records and dataset loading, plus the QASC converter
  prompts.py           prompt rendering and response parsing
  manipulations.py     shuffle_words, remove_shared_words, remove_answer_keywords
  config.py            experiment and model configs (Pydantic models over YAML)
  evaluate.py          mc_accuracy, rouge1, exact_match
  results.py           predictions.jsonl, metrics.json and run.json writers
  run.py               the experiment pipeline
  models.py            Hugging Face backend
  restore_word_order.py  the Section 5.1 probe
  cli.py               the transitive-reasoning command
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
