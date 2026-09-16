# Data

| File | Rows | What it is |
|---|---|---|
| `qasc/dev.json` | 926 | The QASC dev split (Khot et al., 2020), reformatted by `transitive-reasoning prepare-qasc`. Used for every QASC experiment. |
| `bamboogle/original.json` | 125 | The Bamboogle questions of Press et al. (2023), unchanged apart from key names. |
| `bamboogle/with_facts.json` | 112 | This paper's manual decomposition of 112 Bamboogle questions into two facts and a deduction (Section 3.2). Used for every Bamboogle experiment. |
| `bamboogle/gibberish.json` | 112 | `with_facts.json` with names, dates and numbers in the answers turned into gibberish (Section 7.2). The gold `answer` is the gibberish form; `original_answer` keeps the real one. |
| `bamboogle/demonstrations.json` | 3 | The three in-context examples baked into every Bamboogle prompt. None of them is among the 112 test questions. |

The three QASC demonstrations come from the QASC train split and are baked into the prompt files under `prompts/qasc/`.

## Record schema

Every file is a JSON array of objects. Keys:

| Key | Type | Present in |
|---|---|---|
| `question` | string | all files |
| `choices` | string, e.g. `"(A) sand (B) occurs over a wide range ..."` | QASC |
| `fact1`, `fact2` | string | all except `original.json` |
| `deduction` | string; QASC deductions start with "Therefore," | all except `original.json` |
| `answer` | string; QASC form is `"(F) local weather conditions"` | all files |
| `original_answer` | string | `gibberish.json` only |

The loader assigns ids `<dataset>-<index>` (for example `qasc-0007`) in file order.

## Licences and attribution

- QASC is released by the Allen Institute for AI under CC BY 4.0: https://allenai.org/data/qasc
- Bamboogle was released with the self-ask repository by Press et al.: https://github.com/ofirpress/self-ask
- The re-annotated files `with_facts.json`, `gibberish.json` and `demonstrations.json` were created for this paper and are released under CC BY 4.0.
