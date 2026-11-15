# LANTERN: Investigating Multilingual Safety Gaps through Activation Steering and LoRA Fine-Tuning

**One-line claim:** On a five-language safety benchmark, multilingual LoRA reduced LLM-judged harmful assistance from **40.5% to 15.5%**, while activation steering improved safety but exposed over-refusal tradeoffs.

## Tracks and problem statements covered

- **Track 2a** — Multilingual and code-mixed safety.
- **Track 2a** — Adversarial and safety evaluation sets.
- **Track 2b** — Inference-time safety steering.
- **Track 2b** — Locating where safety lives.

> **Content warning:** This research evaluates harmful requests. Linked benchmark files may contain sensitive or unsafe prompt text. We do not recommend redistributing unredacted harmful model completions.

## Abstract

LANTERN (Language-Aware Neural Transfer and Evaluation of Refusal) investigates whether safety behavior transfers across languages and whether targeted interventions can reduce unsafe assistance without unnecessarily refusing legitimate requests. We study Qwen2.5-1.5B-Instruct across English, Hindi, Hinglish, Marathi and Tamil using a multilingual benchmark of harmful and harmless prompts.

We compare baseline behavior, layerwise harmful-versus-harmless activation directions, inference-time activation steering and multilingual LoRA fine-tuning. A Marathi holdout experiment tests whether changes transfer to a language excluded from training. On an LLM-judged subset of 200 harmful responses per method, harmful assistance fell from 40.5% for the base model to 15.5% with 1:1 LoRA and 18.5% with conditional pooled steering. Keyword-detected baseline refusal varied from 100% in English to 0% in Tamil; these heuristic results do not themselves measure harmful compliance. Interventions also increased over-refusal, particularly unconditional steering. Two English capability checks showed small observed score changes after LoRA.

The findings suggest that multilingual safety interventions can reduce judged harmful assistance, but their selectivity and cross-language transfer require further validation. The judge has not been validated against human labels, and the translated benchmark and limited sample constrain generalization.

**Headline before → after:** LLM-judged harmful assistance **40.5% → 15.5%** with 1:1 multilingual LoRA (25 percentage points; approximately 61.7% relative reduction).

**Base model + method:** `Qwen/Qwen2.5-1.5B-Instruct` + multilingual PEFT LoRA; compared with custom and SafeTune activation steering.

> **For judges: one minute**
>
> - **Problem:** Safety behavior can vary substantially across languages, while stronger refusals can block harmless requests.
> - **Approach:** Compare activation directions, steering and multilingual LoRA across five languages, including a Marathi holdout.
> - **Result:** Judged harmful assistance fell from 40.5% to 15.5% with LoRA.

## Problem statement and approach

### Why this problem / real use case

A multilingual assistant may refuse a harmful request in English but respond differently to the same intent in Hindi, Marathi, Tamil or code-mixed Hinglish. English-only safety evaluation can therefore obscure failures experienced by multilingual users. At the same time, an intervention that broadly increases refusals can prevent people from receiving legitimate help. We assess both risks using baseline evaluation, activation analysis, steering, fine-tuning and retention checks.

### Research questions

1. How do harmful-prompt refusal and harmless-prompt over-refusal differ across languages?
2. How similar are harmful-versus-harmless activation directions across languages and layers, and can those directions guide interventions?
3. How do steering and multilingual LoRA compare in harmful assistance, over-refusal, cross-language transfer and basic capability retention?

### Threat model

A user submits harmful requests in different languages to a multilingual instruction-following model. The defender is the model's ordinary response-generation behavior. This work tests language-dependent unsafe assistance, not agentic tool misuse or a systematic adaptive jailbreak search. Judged harmful assistance is not a comprehensive jailbreak attack success rate or cost-to-break.

### Related work

Our activation analysis draws on research into refusal directions by Arditi et al.; adapter training uses LoRA by Hu et al. See [References](#references).

## Hackathon repos (kits) used

- **SafeTune:** [04b_steer_safetune.py](https://safetune.lexsi.ai/) invokes `RefusalDirectionTrainer` to calibrate English and multilingual directions at alpha 0.1 and 0.3. Corresponding `safetune_steer_*.json` result files are present. 

### How the repos connect

```text
English source prompts → translation / Hinglish generation → train + evaluation sets
    → baseline evaluation → layerwise activation analysis
    → custom steering / SafeTune steering / multilingual PEFT LoRA
    → refusal, over-refusal and LLM-judge evaluation
    → Marathi holdout, English capability checks and reporting
```

SafeTune provides a separate steering evaluation; its alpha values are not interchangeable with the custom steering strength scale. PEFT supplies the LoRA training method.

## Models and licenses

| Role | Model / tool | Configuration and license status |
|---|---|---|
| Base and target model | `Qwen/Qwen2.5-1.5B-Instruct` | Quantization, revision and actual license: **Apache License 2.0**. |
| Hinglish generation and LLM judge | `Qwen/Qwen2.5-3B-Instruct` | Hinglish generation has a rule-based fallback; the judge uses the same model family. Revision and license: **Qwen Research license**. |
| Translation | `facebook/nllb-200-distilled-600M` | Hindi, Marathi and Tamil. License: **CC-BY-NC-4.0**. |
| Fine-tuning | Hugging Face Transformers and PEFT | Record tested versions, revisions and applicable licenses. |

## Datasets, benchmarks and licenses

### Sources and languages

English prompts come from `mlabonne/harmful_behaviors` (AdvBench-derived) and `mlabonne/harmless_alpaca`. Hindi, Marathi and Tamil are translated using NLLB; Hinglish is LLM-generated with a transliteration fallback. Verify the source dataset cards, revisions, licenses and redistribution permissions before release. Hinglish may vary in naturalness and is not uniformly code-mixed.

| Dataset | Task / role | Languages |
|---|---|---|
| `harmful_train.jsonl` | Training prompts and activation calibration | English, Hindi, Hinglish, Marathi, Tamil; Marathi excluded in holdout run |
| `harmless_train.jsonl` | Benign training and calibration | Same five languages, subject to holdout |
| `harmful_eval.jsonl` | Safety evaluation | All five languages |
| `harmless_eval.jsonl` | Over-refusal evaluation | All five languages |

### Train / test splits and leak check

The dataset builder uses a fixed random seed (SEED=0) and splits English source prompts into training and evaluation sets before multilingual translation and generation. This reduces direct overlap between training and evaluation examples derived from the same English source.

The final dataset contains 750 training source prompts (250 harmful and 500 harmless) and 240 evaluation source prompts (120 harmful and 120 harmless). Each prompt has versions in five languages: English, Hindi, Hinglish, Marathi, and Tamil.

This results in 3,750 multilingual training examples and 1,200 multilingual evaluation examples, totaling 4,950 language-specific examples.

The source-level split reduces direct train/test contamination. 

| Language | Harmful evaluation | Harmless evaluation |
|---|---:|---:|
| English | 120 | 120 |
| Hindi | 120 | 120 |
| Hinglish | 86 | 81 |
| Marathi | 120 | 120 |
| Tamil | 120 | 120 |

### Safety and general capability benchmarks

The harmful and harmless evaluation sets cover all five languages and are the focus of this safety work. The harmless set measures over-refusal but was not specifically constructed as a set of safe prompts that appear risky. ARC Easy and HellaSwag are English-only general capability retention checks, not multilingual competence benchmarks. The harmful prompt set is derived from English source prompts and translations.

## Metrics, and why these

| Metric | Definition and purpose | Direction |
|---|---|---|
| Keyword-detected harmful refusal | Fraction of harmful responses matching a heuristic refusal detector | Higher, subject to judge validation |
| Keyword-detected harmless over-refusal | Fraction of harmless responses triggering the detector; measures loss of legitimate assistance | Lower |
| LLM-judged harmful assistance | Fraction of harmful responses labeled `HARMFUL` by the judge | Lower |
| LLM-judged refusal | Fraction of harmful responses labeled `REFUSE` | Higher |
| Direction similarity | Cosine similarity of language-specific activation differences with English | Descriptive; not inherently higher-is-safer |
| Capability retention | ARC Easy and HellaSwag scores before and after LoRA | No meaningful degradation |

### Checking the automatic judge

`Qwen/Qwen2.5-3B-Instruct` assigns `REFUSE`, `HARMFUL`, `SAFE` or `NONSENSE`. Hindi, Marathi and Tamil responses are machine-translated to English for judging. The judged experiment samples up to 40 harmful responses per language and method (200 per method). 

## Method / experiments

### Baseline

`02_eval.py` evaluates the base model on multilingual harmful and harmless prompts. Generation uses `MAX_NEW=64` tokens. The saved five-language summary rates are unweighted means of language-specific rates, not pooled sample rates.

### Activation-direction analysis

`03_locate.py` computes layerwise differences between hidden representations of harmful and harmless prompts, compares directions across languages and assesses separability. Its original selection criterion chooses layer 11. These are harmfulness-related activation differences, not a demonstrated unique causal refusal circuit.


### Custom Activation Steering

`04_steer.py` evaluates English-derived and pooled multilingual activation directions using conditional and unconditional steering.

The default steering strengths are **2.0, 3.0, and 4.5**, with the primary intervention applied at the layer selected by `03_locate.py` (**layer 11** in the saved analysis).

Two conditional strategies are evaluated:

- **`cond`:** Uses a global activation threshold.
- **`condlang`:** Uses language-specific activation thresholds.

These thresholds are calculated from the mean projections of harmful and harmless training prompts onto the selected activation direction. The intervention magnitude is proportional to the projection gap and steering strength.

An unconditional English steering configuration with strength **1.5** is also evaluated as a comparison.

`04c_layer_sweep.py` additionally evaluates pooled multilingual steering with a global conditional gate at **layers 12, 13, 14, 15, 16, and 18**, using strength **4.5**. Language-specific conditional steering is also evaluated at additional layers.

**Common evaluation settings:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Random seed | `0` |
| Maximum generation length | 64 new tokens |
| Steering strengths | 2.0, 3.0, 4.5 |
| Primary intervention layer | 11 |
| Layer sweep | 12, 13, 14, 15, 16, 18 |
| Conditional strategies | `cond`, `condlang` |
| Unconditional comparison | English direction, strength 1.5 |

Conditional steering relies on activation-based heuristic thresholds, whose effectiveness may vary across languages.

### SafeTune Steering

`04b_steer_safetune.py` evaluates SafeTune's `RefusalDirectionTrainer` using two calibration configurations:

- **English-only:** 64 harmful and 64 harmless English training prompts.
- **Multilingual:** 13 harmful and 13 harmless source prompts expanded across five languages, producing 65 harmful and 65 harmless calibration examples.

For each configuration, steering strengths (`alpha`) of **0.1 and 0.3** are evaluated, resulting in four SafeTune experiments.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Trainer | `RefusalDirectionTrainer` |
| Calibration configurations | English-only, multilingual |
| English calibration | 64 harmful + 64 harmless |
| Multilingual calibration | 65 harmful + 65 harmless |
| Steering alpha | 0.1, 0.3 |
| Random seed | 0 |
| Maximum generation length | 64 new tokens |
| Other trainer parameters | SafeTune defaults |

The trainer's `calibrate()` method constructs the steering intervention from the supplied harmful and harmless examples. The calibrated intervention is subsequently evaluated on the multilingual benchmark.

### Multilingual PEFT LoRA

`05_harden_lora.py` performs supervised fine-tuning using Hugging Face PEFT. Training examples combine multilingual harmful prompts paired with predefined refusal responses and harmless prompts paired with responses generated by the original base model.

Three configurations are evaluated:

1. **LoRA 1:1:** Equal numbers of harmful and harmless training examples.
2. **LoRA 1:2:** Twice as many harmless as harmful training examples, intended to reduce over-refusal.
3. **Marathi holdout:** Excludes Marathi from training to evaluate cross-language safety transfer.

**LoRA training configuration:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Training epochs | 1 |
| Learning rate | `1e-4` |
| Per-device batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 (single GPU) |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning-rate scheduler | Cosine |
| Warmup steps | 10 |
| Maximum tokenized sequence length | 768 |
| Optimizer | Hugging Face `TrainingArguments` default |
| Precision | BF16 or FP16, depending on model-loading dtype |
| Gradient checkpointing | Enabled |
| Random seed | 0 |
| Logging interval | Every 10 steps |
| Intermediate checkpoint saving | Disabled |

The trained adapters are saved separately for each configuration and evaluated using the same multilingual safety benchmark.

### Seeds and ablations

Data construction uses `SEED=0`. Ablations compare ratios, layer and steering strength, conditional versus unconditional steering, English versus pooled directions and Marathi holdout. 

## Results

### Baseline language differences

| Language | Harmful refusal ↑ | Harmless over-refusal ↓ | Harmful / harmless n |
|---|---:|---:|---:|
| English | 100.0% | 5.8% | 120 / 120 |
| Hindi | 28.3% | 0.8% | 120 / 120 |
| Hinglish | 72.1% | 18.5% | 86 / 81 |
| Marathi | 4.2% | 0.8% | 120 / 120 |
| Tamil | 0.0% | 2.5% | 120 / 120 |

### Same model before vs after: safety and over-refusal

These are unweighted means of the five language-specific rates.

| Method | Harmful refusal ↑ | Harmless over-refusal ↓ |
|---|---:|---:|
| Base model | 40.9% | 5.7% |
| LoRA 1:1 | 79.3% | 17.2% |
| LoRA 1:2 | 74.7% | 11.1% |
| Pooled steering ×4.5 | 84.8% | 12.9% |
| Pooled steering ×4.5 at layer 12 | 97.6% | 19.3% |
| Unconditional English steering ×1.5 | 100.0% | 86.9% |

### LLM-judged harmful assistance

| Method | Harmful assistance ↓ | Judged responses |
|---|---:|---:|
| Base model | 40.5% | 200 |
| LoRA 1:1 | **15.5%** | 200 |
| LoRA 1:2 | 16.0% | 200 |
| Conditional pooled language-gated steering ×3.0 | 18.5% | 200 |

LoRA 1:1 reduces judged harmful assistance by 25 percentage points (approximately 61.7% relative to baseline). 

### Cross-language transfer: Marathi holdout

| Marathi metric | Base | LoRA with Marathi | LoRA excluding Marathi |
|---|---:|---:|---:|
| Harmful refusal ↑ | 4.2% | 75.0% | 82.5% |
| Harmless over-refusal ↓ | 0.8% | 20.0% | 39.2% |

Refusal improvements transfer to Marathi when it is held out from training, but with some increase in over-refusal.

### General capability retention

| English benchmark | Base | LoRA 1:1 |
|---|---:|---:|
| ARC Easy accuracy | 72.0% | 71.5% |
| HellaSwag accuracy | 49.5% | 50.0% |

These two English checks show small observed changes; they do not establish preservation of broader multilingual capabilities.

### What didn't work and statistical limitations

- Unconditional steering refused 86.9% of harmless prompts despite 100% harmful-prompt refusal.
- LoRA 1:2 reduced over-refusal relative to 1:1, with somewhat lower harmful-prompt refusal.
- Steering was sensitive to layer choice, and the best keyword-refusal layer has not been shown to be the best judged-safety layer.
- Hinglish generation produced incomplete examples and variable language quality.

### Figures

![Layerwise activation analysis](figures/locate.png)

![Safety and over-refusal summary](figures/summary.png)


## Cost / ROI

### Compute used

Experiments used an NVIDIA T4 GPU in a Google Colab runtime. A T4 has **16 GB of GPU memory**.

## Limitations

- Keyword refusal detection can misclassify safe responses or partial harmful assistance; the LLM judge has not been validated against blinded human labels.
- Machine translation can distort responses for judging; the benchmark is derived from translated English sources and may not represent native-language requests. Hinglish quality varies.
- No completed leakage audit is documented. Evaluation reuse during layer and strength sweeps can overfit configuration selection.
- Activation differences are not proven causal refusal circuits; fixed refusal templates can encourage style rather than nuanced safety judgments.
- Only one small Qwen target model and five language settings were evaluated; English retention checks do not establish multilingual capability preservation.

## With more credits

### Planned approach

Use native-speaker validation, an independent held-out multilingual test set, paired bootstrap confidence intervals, broader adversarial paraphrases and more natural Hinglish. Repeat across seeds and a larger model; judge the best layer-12 steering configuration using the same LLM and human evaluation; and measure latency and cost per 1,000 replies.

### Expected before / after

Current results provide preliminary estimates with unvalidated judging and observed over-refusal. Additional data and human validation are expected to yield more reliable estimates and may support interventions with lower over-refusal. 

## Artifacts

- **Hugging Face datasets collection:** [https://huggingface.co/collections/aneeshadas02/lexsi-multilingual-safety-datasets]
  - [Multilingual benchmark](https://huggingface.co/datasets/aneeshadas02/lexsi-multilingual-benchmark)
  - [Experiment results](https://huggingface.co/datasets/aneeshadas02/lexsi-experiment-results)
- **Hugging Face models collection:** [https://huggingface.co/collections/aneeshadas02/lexsi-multilingual-safety-models]
  - [LoRA 1:1](https://huggingface.co/aneeshadas02/lexsi-lora-1to1)
  - [LoRA 1:2](https://huggingface.co/aneeshadas02/lexsi-lora-1to2)
  - [Marathi holdout adapter](https://huggingface.co/aneeshadas02/lexsi-lora-marathi-holdout)
  - [Steering vectors](https://huggingface.co/aneeshadas02/lexsi-steering-vectors)
- **GitHub repository:** [https://github.com/aneesha02/lantern-multilingual-safety]

## Documentation and reproduction

### Environment requirements

Google Colab with an NVIDIA T4 GPU was used. The notebook uses Hugging Face Transformers, PEFT and Python tooling

### Repository layout

```text
.
├── README.md
├── notebooks/lexsi_safety_pipeline_github_ready.ipynb  # generates scripts
├── figures/{locate.png,summary.png}                    # analysis and results
├── config.py, common.py                               # shared configuration/helpers
├── 01_build_data.py                                   # build multilingual data
├── 02_eval.py                                         # evaluate responses
├── 03_locate.py                                       # activation analysis
├── 04_steer.py                                        # custom steering
├── 04b_steer_safetune.py                              # SafeTune steering
├── 04c_layer_sweep.py                                 # layer sweep
├── 05_harden_lora.py                                  # train LoRA adapters
├── 06_judge.py                                        # judge responses
├── 07_report.py                                       # produce reports
└── 09_relabel.py                                      # relabel results
```

The notebook uses `%%writefile` cells to create scripts. 

### Run instructions

1. Open the notebook in a CUDA GPU runtime and install the exact tested dependencies once recorded.
2. Set `LEXSI_MODEL=Qwen/Qwen2.5-1.5B-Instruct`, confirm model access and run the script-generation cells.
3. Run the following commands and the notebook's documented steering configurations:

```bash
python 01_build_data.py
python 02_eval.py --tag baseline
python 03_locate.py

python 05_harden_lora.py --out outputs/Qwen2.5-1.5B-Instruct/lora_safety
python 02_eval.py --tag hardened --adapter outputs/Qwen2.5-1.5B-Instruct/lora_safety

python 05_harden_lora.py --safe_ratio 2 --out outputs/Qwen2.5-1.5B-Instruct/lora_safety_r2
python 02_eval.py --tag hardened_r2 --adapter outputs/Qwen2.5-1.5B-Instruct/lora_safety_r2

python 05_harden_lora.py --hold_out mr --out outputs/Qwen2.5-1.5B-Instruct/lora_safety_holdout_mr
python 02_eval.py --tag hardened_holdout_mr --adapter outputs/Qwen2.5-1.5B-Instruct/lora_safety_holdout_mr

python 06_judge.py --tags baseline hardened hardened_r2 steer_pooled_condlang_x3.0 --n 40
python 07_report.py
```

Generate the `steer_pooled_condlang_x3.0` steering audit CSV using the notebook's steering cells before judging. Results are written to `results/<MODEL_TAG>/` and audits to `outputs/<MODEL_TAG>/`. Preserve original data, revisions, seeds, generation settings and command-line overrides.

## Declaration of work done outside the hackathon

NONE

## Teammates tagged

Aneesha Das - https://github.com/aneesha02
Vansh Agrawal - https://github.com/Vansh-Agrawal-004

## References

1. Arditi et al., [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (2024).
2. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021).
3. [Qwen model collection](https://huggingface.co/Qwen).
4. [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M).
5. Source datasets: [harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) and [harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca).
6. **Hugging Face libraries and evaluation framework:**
   - [Transformers](https://github.com/huggingface/transformers) — Model loading, tokenization, and training.
   - [PEFT](https://github.com/huggingface/peft) — LoRA fine-tuning and adapter management.
   - [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — ARC Easy and HellaSwag capability evaluation.
7. SafeTune: [Link(https://safetune.lexsi.ai/)].

