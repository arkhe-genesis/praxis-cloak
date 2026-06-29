---
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- lora
- peft
- qwen
- privacy
- pii
- on-device
---

> **Improved using Qwen** — this is a derivative fine-tune of Qwen/Qwen2.5-3B-Instruct, distributed under the Qwen Research License (non-commercial).

# praxis/relevance-3b: Keep-vs-Scrub Decision Judge for Disclosure-Control Pipelines

**praxis/relevance-3b** is a lightweight, fine-tuned Qwen2.5-3B-Instruct LoRA model that makes *question-aware* keep-vs-scrub decisions on detected PII (locations, organizations). Given a user message, the question it contains, and a list of detected PII entities, the model judges whether each entity is *needed to answer the question* (KEEP) or *incidental* (SCRUB, replace with a fake or generalization).

It is designed as the relevance stage in a disclosure-control pipeline: detect PII → **judge relevance** → substitute → cloud → rehydrate. It replaces keyword heuristics that over-scrub ~93% of load-bearing entities, which would cause nonsensical ("lobotomized") answers.

**Important framing:** This model provides *disclosure control*, not a privacy guarantee. A kept location still goes to the cloud unmodified; privacy depends on the downstream substitution layer. Use only inside the full detect→substitute→cloud→rehydrate pipeline.

## Model Details

### Model Description

- **Base Model:** Qwen/Qwen2.5-3B-Instruct
- **Model Type:** LoRA-adapted causal language model
- **Task:** Per-entity relevance judgment (KEEP/SCRUB binary classification)
- **Language(s):** English
- **Fine-tune Method:** LoRA (r=16, α=32, dropout=0)
- **License:** Qwen Research License (non-commercial; commercial use requires a separate license from Alibaba Cloud)
- **Derivative Notice:** This is a derivative work improving the Qwen model. Distributions must display "Built with Qwen" or "Improved using Qwen" in product documentation per the Qwen Research License.

### Model Sources

- **Base Model Repository:** [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **HuggingFace:** https://huggingface.co/praxis-nation/relevance-3b
- **Ollama:** `ollama create praxis/relevance-3b -f Modelfile`
- **Training Code:** github.com/praxis-society/praxis-cloak/evals/train_relevance_gpu.py
- **Training Dataset (primary):** [CAPID](https://github.com/MariaPonomarenko38/CAPID) (MIT License)

## Uses

### Direct Use

Use case: make question-aware keep/scrub decisions on PII entities within a disclosure-control pipeline.

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import re

model_id = "praxis/relevance-3b"
model = AutoPeftModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

message = "I'm moving from Toronto to London for a job at Accenture. What's the cost of living there?"
question = "What's the cost of living there?"
details = ["Toronto", "London", "Accenture"]

system = """You decide which personal details in a message are NEEDED to answer the user's question well. \
A detail is RELEVANT only if answering the question genuinely depends on that specific detail \
(e.g. a location when the answer depends on local law or availability). \
It is NOT relevant if it is incidental background the answer does not need."""

detail_str = "\n".join(f"{i+1}. {d}" for i, d in enumerate(details))
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": f"Message: {message}\nQuestion: {question}\nDetails:\n{detail_str}"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256, temperature=0)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
# Expected:
# 1. RELEVANT  (London is needed — the question is about cost of living *there*)
# 2. RELEVANT  (destination of relocation)
# 3. NOT       (Accenture is incidental — cost of living depends on location, not employer)
```

**Ollama (quantized, on-device):**

```bash
ollama create praxis/relevance-3b -f Modelfile
ollama run praxis/relevance-3b \
  "Message: I work at Microsoft in Seattle. Is it a good place to raise a family?
Question: Is it a good place to raise a family?
Details:
1. Microsoft
2. Seattle"
# Expected:
# 1. NOT       (employer is incidental; the question is about the location)
# 2. RELEVANT  (location is the subject of the question)
```

### Downstream Use

- **Disclosure-control pipelines:** Core stage in detect→substitute→cloud→rehydrate architectures. The relevance judge decides what to scrub so the substitution layer acts on the right entities.
- **Utility preservation:** By judging relevance rather than using blanket rules, the model preserves load-bearing details and prevents lobotomized answers.
- **On-device deployment:** Runs locally; keep/scrub decisions are never sent to the cloud.

### Out-of-Scope Use

- **Not a privacy guarantee.** A kept location still goes to the cloud unmodified. Privacy depends on the downstream substitution layer.
- **Not for fine-grained PII classification.** The model judges relevance but does not classify PII into fine types (medical, financial, biometric). It operates on coarse categories (location, organization).
- **Not for re-identification prevention in isolation.** Re-id risk is measured on the full pipeline (detect→substitute→cloud→rehydrate).

## Bias, Risks, and Limitations

### Model Limitations

1. **~7% over-scrubbing on real prompts:** On real customer-distribution WildChat prompts (350 samples), the model scrubs ~7% of location entities that should have been kept (false-scrub / "lobotomy"). The target is ≤5%; this gap is near the human-ceiling on the task (see Gold Audit below).

2. **~69% of kept incidental locations escape to the cloud:** These are caught downstream by the substitution layer (generalized or scrubbed), so the effective privacy risk is post-substitution, not raw false-keep.

3. **Accuracy ceiling ~84–86%:** On the held-out CAPID dataset, accuracy plateaus at 84% on locations. This is due to ~9% gold label noise in CAPID (confirmed by independent audit) and inherent task ambiguity (human raters agree only 66–72% on borderline keep/scrub calls). The accuracy ceiling is data-bound, not capacity-bound.

4. **Multi-location referent resolution is hard:** Questions like "comparable compensation *there*?" with multiple locations (origin vs. destination) are systematically mishandled. The model sometimes resolves the pronoun to the wrong referent.

5. **Binary only:** The model outputs KEEP or SCRUB, not fine-grained actions like "keep the type, drop the identity" (e.g., "Lisbon" → "a European city"). Generalization is handled by the substitution layer.

6. **Limited to locations and organizations:** Relevance judgment on other PII types (names, dates, amounts, emails) is not implemented in this model.

### Training Data and Synthetic Label Risk

- **Train data:** 1,897 real + cleaned CAPID examples + 800 synthetic (Praxis-generated, no public data)
- **Synthetic-label circularity:** ~30% of training examples use model-generated labels; the model partly imitates a model's notion of relevance rather than ground truth. This is a known bias.
- **Not published:** Training data is not public (CAPID contains real PII; synthetic data is internal). Only weights are distributed.

### Bias and Fairness

- Training data skews toward relocation/jurisdiction questions (CAPID corpus). Real prompts show different skew (location entities are usually the subject of the question, not incidental context). The model generalizes reasonably (7% over-scrub on realdist), but this is a known distribution shift.
- Biases in entity coverage: Western company names, English-speaking geographies over-represented.
- No debiasing applied.

### Recommendations

1. **Always pair with a substitution/generalization layer.** Kept-incidental locations must be generalized before cloud transmission. This is required for privacy; the keep/scrub decision alone is not sufficient.

2. **Validate on your data distribution.** Test on real questions in your domain before deployment.

3. **Measure end-to-end disclosure impact.** Use re-identification or membership-inference testing on the full pipeline, not the model in isolation.

4. **Display "Improved using Qwen" in product docs.** Required by the Qwen Research License.

## How to Get Started

### HuggingFace / Transformers

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import re

model_name = "praxis/relevance-3b"
model = AutoPeftModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

system = """You decide which personal details in a message are NEEDED to answer the user's question well. \
A detail is RELEVANT only if answering the question genuinely depends on that specific detail \
(e.g. a location when the answer depends on local law or availability). \
It is NOT relevant if it is incidental background the answer does not need."""

message = "I'm considering moving to Japan and wondering about the cost of living there compared to my current city, Toronto."
question = "How does cost of living in Japan compare to Toronto?"
details = ["Japan", "Toronto"]

detail_str = "\n".join(f"{i+1}. {d}" for i, d in enumerate(details))
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": f"Message: {message}\nQuestion: {question}\nDetails:\n{detail_str}"}
]

text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256, temperature=0)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

matches = re.findall(r'(\d+)\.\s*(RELEVANT|NOT)', generated, re.IGNORECASE)
for idx, relevance in matches:
    span = details[int(idx) - 1]
    decision = "KEEP" if "RELEVANT" in relevance.upper() else "SCRUB"
    print(f"{span}: {decision}")
# Output:
# Japan: KEEP
# Toronto: KEEP
```

### Ollama (Quantized)

```bash
# Default: q4_k_m (~1.8 GB, on-device)
ollama create praxis/relevance-3b -f Modelfile
ollama run praxis/relevance-3b \
  "Message: I work at Google in San Francisco. How's the weather?
Question: How's the weather?
Details:
1. Google
2. San Francisco"

# Reference quality: q8_0 (~3.1 GB)
ollama create praxis/relevance-3b -f Modelfile
```

## Training Details

### Training Data

- **Primary source:** CAPID dataset (https://github.com/MariaPonomarenko38/CAPID, MIT License)
  - 1,897 training examples from CAPID's real Reddit relocation/advice posts
  - Each example: context + question + per-detail relevance label (KEEP=1, SCRUB=0)
  - Test/eval splits held out: CAPID test.jsonl (200) + reddit.jsonl (149) — never trained on
- **Secondary source:** Synthetic data (Praxis-generated, 800 examples via `evals/build_relevance_synth.py`)
  - Targets measured failure modes: multi-location relocation discrimination, comparison questions requiring both places
  - All synthetic entities are invented; no real PII
- **Composition:** 2,697 total train examples (1,897 real + 800 synthetic)
- **Note:** Raw training data is NOT published. Only weights are distributed.

### Training Procedure

- **Framework:** Unsloth (QLoRA, 4-bit, bf16)
- **Base Model:** Qwen2.5-3B-Instruct
- **LoRA Config:** r=16, α=32, dropout=0, target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Optimizer:** AdamW 8-bit, weight decay 0.01
- **Learning Rate:** 1e-4, linear schedule
- **Epochs:** 3
- **Batch:** 4 per-device, 4 gradient accumulation (effective 16)
- **Warmup:** 5% of steps
- **Loss:** Response-only (system + user masked)
- **Eval Strategy:** Every 50 steps on held-out CAPID dev.jsonl (210 examples)
- **Hardware:** Single H200 GPU (Nebius rented, ~1.5h runtime, ~$5–6)

### Sizes

- **Model size (adapter):** ~5 MB (LoRA weights only)
- **Merged model (bf16):** ~6.5 GB
- **GGUF quantizations:** q4_k_m ~1.8 GB, q8_0 ~3.1 GB
- **Inference speed (q4_k_m, Apple Silicon):** ~30–100 ms per judgment

## Evaluation

### Results

| Benchmark | Location Over-Scrub | Location Accuracy | Org Over-Scrub | Org Accuracy | Notes |
|---|---|---|---|---|---|
| **CAPID raw** | 10% | 84% | 7% | 94% | Original; ~9% label noise |
| **CAPID clean** | 10% | 84% | 8% | 95% | Audited gold; ~91% perfect ceiling |
| **Realdist (real)** | **7%** | **86%** | 12% | — | Real customer distribution; primary benchmark |
| **PIIBench** | 37% | 78% | 20% | — | Structured format; out-of-distribution |

**E2E Validation (60 test cases: 40 load-bearing + 20 incidental locations, full pipeline)**

| Metric | Raw Prompt | Judge ON | Judge OFF |
|---|---|---|---|
| **Usefulness (Likert 1–5)** | 3.92 | **3.70** | 2.60 |
| **Must-keep survival** | — | **93%** | 52% |
| **Must-hide scrubbed** | — | 42% | 60% |
| **Re-id mean rank** | 1.73 | **1.72** | 1.72 |

**Interpretation:**
- **GATE 1 (no lobotomy) PASS:** Judge keeps 93% of load-bearing locations vs keyword gate's 52%; usefulness drops only 5% vs raw (3.70 vs 3.92), far better than judge-OFF (2.60).
- **GATE 2 (privacy not raised) PASS:** Despite keeping more incidental locations, re-id does not rise (1.72 = raw). The substitution layer successfully generalizes kept-incidental locations; privacy is preserved downstream.

### Alternative Models Tested

| Model | Loc Over-Scrub | Org Over-Scrub | Note |
|---|---|---|---|
| **praxis/relevance-3b (3B)** | **7%** | 12% | **WINNER** — best on realdist; deployable on-device |
| FT-7B recipe | 8% | **5%** | Relay-tier alternative; worse over-scrub on realdist |
| FT-14B recipe | 9% | 6% | Worse over-scrub; not recommended (deployment cost) |
| Base 14B prompt-only | 20% | 41% | Unacceptable; why fine-tuning is needed |

Bigger base models do not improve location over-scrub on the real distribution. The ceiling is data- and gold-bound, not capacity-bound. 3B is the optimal choice for on-device deployment.

### Gold Audit and Inter-Rater Ceiling

- Raw CAPID gold vs independent re-label: 83.9% agreement; 63 disagreements (16.1%)
- High-confidence gold errors: 24 clear flips
- Conservative cleaned gold: 15 triple-confirmed flips
- **Inter-rater ceiling (88 borderline spans, 4 raters):** 72.2% mean pairwise agreement (κ=0.44) on borderline spans; 87.3% on clear cases
- **Realistic accuracy ceiling:** ~84% exact-match population-weighted — the ≥95% bar sits above the human ceiling

## Technical Specifications

- **Base:** Qwen2.5-3B-Instruct (3B parameters, transformer, causal LM)
- **Adapter:** LoRA, r=16, α=32; ~67M trainable params (0.4% of base)
- **Precision:** bf16 (merged); quantized to q8_0, q4_k_m for deployment
- **Minimum inference (CPU, q4_k_m):** Apple Silicon (M1+), x86-64 (Ryzen 5000+, Intel 12th gen+)
- **Ollama:** Runs on macOS, Linux, Windows

## Citation

```bibtex
@misc{relevance-3b,
  title={praxis/relevance-3b: Question-Aware Keep-vs-Scrub Decision Judge for Disclosure-Control Pipelines},
  author={Praxis},
  year={2026},
  url={https://huggingface.co/praxis-nation/relevance-3b},
  note={Fine-tuned Qwen2.5-3B-Instruct LoRA. Base model license: Qwen Research License (non-commercial). Training uses CAPID dataset (MIT License, github.com/MariaPonomarenko38/CAPID).}
}
```

## Glossary

- **LoRA:** Low-Rank Adaptation; parameter-efficient fine-tuning via small trainable adapters
- **QLoRA:** LoRA with 4-bit quantization; reduces VRAM during training
- **GGUF:** Quantized model format optimized for CPU/edge inference (llama.cpp, Ollama)
- **PII:** Personally Identifiable Information (names, locations, organizations, emails, phone numbers)
- **Over-scrub:** Scrubbing a relevant (needed) entity; causes lobotomized (nonsensical) answers
- **False-keep:** Keeping an irrelevant entity; privacy leak if no downstream generalization
- **CAPID:** Context-Aware PII Detection dataset (github.com/MariaPonomarenko38/CAPID, MIT License)
- **Realdist:** Real-customer-distribution benchmark derived from allenai/WildChat (AI2 ImpACT license)
- **Disclosure control:** Reducing PII surface area in transmitted data; distinct from a privacy guarantee

## Framework Versions

- PEFT: 0.19.1
- Transformers: 4.48+
- Torch: 2.1+
- Unsloth: Latest (github.com/unslothai/unsloth)
- TRL: 0.13+

## Disclaimer

This model is a research artifact. It is provided as-is without warranty. **Use only as part of a full disclosure-control pipeline (detect→substitute→cloud→rehydrate)**, not in isolation. Validate end-to-end on your data distribution before production deployment. Training data is private and not distributed; model weights are redistributable under the Qwen Research License (non-commercial).

---

*Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.*
*See https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE*

*CAPID dataset: MIT License. Copyright (c) MariaPonomarenko38 and contributors. https://github.com/MariaPonomarenko38/CAPID*
