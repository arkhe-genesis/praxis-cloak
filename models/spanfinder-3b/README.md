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

# praxis/spanfinder-3b: PII Span Detection for On-Device Disclosure Control

**praxis/spanfinder-3b** is a lightweight, fine-tuned Qwen2.5-3B-Instruct LoRA model for detecting and extracting personally identifiable information (PII) spans from user text. It is designed for on-device disclosure-control pipelines that scrub sensitive data before sending prompts to cloud language models.

**Important framing:** This model provides *disclosure control*, not a privacy guarantee. It is a single stage in a detect→substitute→rehydrate pipeline. Used in isolation it will leak PII; its recall ceiling (~63%) is a known design parameter, backstopped by the substitution and rehydration layers.

## Model Details

### Model Description

- **Base Model:** Qwen/Qwen2.5-3B-Instruct
- **Model Type:** LoRA-adapted causal language model
- **Architecture:** Qwen2.5-3B (3 billion parameters)
- **Task:** PII span extraction and categorization
- **Language(s):** English
- **Fine-tune Method:** LoRA (r=16, α=32, dropout=0)
- **License:** Qwen Research License (non-commercial; commercial use requires a separate license from Alibaba Cloud)
- **Derivative Notice:** This is a derivative work improving the Qwen model. Distributions must display "Built with Qwen" or "Improved using Qwen" in product documentation per the Qwen Research License.

### Model Sources

- **Base Model Repository:** [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **HuggingFace:** https://huggingface.co/praxis-nation/spanfinder-3b
- **Ollama:** `ollama pull hf.co/praxis-nation/spanfinder-3b:Q4_K_M`
- **Training Code:** github.com/praxis-society/praxis-cloak/evals/train_spanfinder_gpu.py

## Uses

### Direct Use

Use case: extract PII spans from user input *before* passing prompts to cloud LLMs, as part of a disclosure-control pipeline.

```python
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

model_id = "praxis/spanfinder-3b"
model = AutoPeftModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "I live in Toronto and work at Accenture. How do I apply for a visa?"
messages = [
    {"role": "system", "content": "Extract all PII spans from this message. Return each span on a line: span | category"},
    {"role": "user", "content": prompt}
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=256)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)  # Expected: "Toronto | location\nAccenture | organization"
```

**Ollama (quantized, on-device):**

```bash
ollama pull hf.co/praxis-nation/spanfinder-3b:Q4_K_M
ollama run hf.co/praxis-nation/spanfinder-3b:Q4_K_M "Extract all PII from: I live in San Francisco and work at Google."
```

### Downstream Use

- **Disclosure-control pipelines:** Plug into detect→substitute→cloud→rehydrate architectures
- **Data minimization:** Identify and mask/generalize PII before data sharing
- **Compliance auditing:** Detect PII in datasets for privacy impact assessments

### Out-of-Scope Use

- **Not for attribution or re-identification:** This model detects PII but does not prevent re-identification when combined with other data sources. A downstream substitution layer is required.
- **Not a privacy guarantee:** Detection is a best-effort heuristic with known gaps (~37% undetected spans at model level; see Limitations).
- **Not for production in isolation:** Deployed without downstream substitution and rehydration controls, this model will leak sensitive data.

## Bias, Risks, and Limitations

### Model Limitations

1. **Incomplete span detection:** ~37% of PII spans remain undetected at the model level. This is the per-model leakage ceiling; the full pipeline backstops misses via the substitution layer (generalize or mask undetected spans).

2. **Trained on low-PII interview-style text:** Performance on high-PII, free-form user text (narrative personal stories, address-rich forms) has not been benchmarked.

3. **English-only:** No multilingual capability; behavior on non-English text is undefined.

4. **Span-level detection only:** The model extracts text spans but does not classify fine-grained PII types (medical, financial, biometric). Category labels are coarse (name, location, organization, email, phone).

5. **No context-aware relevance filtering:** The model detects PII but does not judge whether a detected span is *needed* to answer a question. A parallel relevance judge (praxis/relevance-3b) handles that decision.

### Bias and Fairness

- Training data is skewed toward English-speaking, Western company names and geographic locations (interview-heavy corpus). Biases in entity frequency and geographic representation are preserved.
- No debiasing applied; model reflects train data skew.

### Recommendations

1. **Always pair with a substitution/generalization layer.** Detected spans must be scrubbed or generalized before any sensitive downstream use.

2. **Validate on your data distribution.** Test on a sample of real user input in your domain before deployment.

3. **Measure end-to-end disclosure impact.** Use re-identification or membership-inference testing on the full pipeline (detect→scrub→cloud→rehydrate), not the model in isolation.

4. **Display "Improved using Qwen" in product docs.** Required by the Qwen Research License.

## How to Get Started

### HuggingFace / Transformers

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_name = "praxis/spanfinder-3b"
model = AutoPeftModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You extract all personally identifiable information (names, places, organizations, emails, phone numbers) from user messages. Return each span on a line: span | category"},
    {"role": "user", "content": "My name is Alice and I live in Paris. I work at Microsoft."}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=512, temperature=0)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# Output: Alice | name\nParis | location\nMicrosoft | organization
```

### Ollama (Quantized)

```bash
# Default: q4_k_m (~1.8 GB, on-device)
ollama pull hf.co/praxis-nation/spanfinder-3b:Q4_K_M
ollama run hf.co/praxis-nation/spanfinder-3b:Q4_K_M "Extract PII: I'm John from Seattle, working at Apple."

# Reference quality: q8_0 (~3.1 GB)
ollama pull hf.co/praxis-nation/spanfinder-3b:Q8_0
```

## Training Details

### Training Data

- **Source:** Real interview QA turns (private Praxis dataset) + synthetic generated examples via `evals/build_spanfinder_data.py`
- **Size:** ~2,500 training examples
- **Note:** Raw training data contains real PII and is **NOT published**. Only the trained weights are distributed. No training data files appear in this repository.

### Training Procedure

- **Framework:** Unsloth (QLoRA, 4-bit, bf16)
- **Base Model:** Qwen2.5-3B-Instruct
- **LoRA Config:** r=16, α=32, dropout=0, target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Optimizer:** AdamW 8-bit, weight decay 0.01
- **Learning Rate:** 1e-4, linear schedule
- **Epochs:** 4
- **Batch:** 8 per-device, 2 gradient accumulation (effective 16)
- **Warmup:** 5% of steps
- **Loss:** Response-only (system + user masked; completion-only loss)
- **Hardware:** Single H200 GPU (Nebius rented; ~1.5h runtime, ~$5–6)
- **Reproducibility:** Script: `evals/train_spanfinder_gpu.py`

### Sizes

- **Model size (adapter):** ~5 MB (LoRA weights only)
- **Merged model (bf16):** ~6.5 GB
- **GGUF quantizations:** q4_k_m ~1.8 GB, q8_0 ~3.1 GB
- **Inference speed (q4_k_m, Apple Silicon):** ~20–50 ms per span extraction

## Evaluation

### Results

| Benchmark | Precision | Recall | F1 | Note |
|---|---|---|---|---|
| Interview (dev) | 91% | 63% | 0.75 | Real, low-PII per-turn |
| Synthetic (test) | 89% | 67% | 0.77 | Controlled span distribution |
| **Per-model leak rate** | — | **63% detected** (37% undetected) | — | Undetected spans handled by substitution layer |

**Interpretation:** The model catches ~63% of PII at detection time. The remaining ~37% is the per-model ceiling — addressed in the full pipeline via substitution (generalize or mask undetected spans). The detector is paired with a fast-scrub keep-gate (regex/keywords) and a downstream substitution layer (realistic fake substitution), which together achieve end-to-end disclosure control without requiring perfect detection.

### Comparison to Base Model

Base Qwen2.5-3B-Instruct (no fine-tuning), prompted for span extraction: ~15% recall, ~10% precision (mostly hallucination). The fine-tune is a ~4–5x recall improvement and ~8–9x precision improvement over base.

### Common Failure Modes

1. **Missed abbreviations:** "NYC" not detected when full name "New York City" is in context
2. **Pronouns + context:** "I'm going there" — "there" not extracted as a location span (resolved at the substitution layer)
3. **Embedded entities:** "John from the San Francisco office at Google" — entity boundary ambiguity

These are accepted limitations; the full pipeline handles them via context-aware substitution.

## Technical Specifications

- **Base:** Qwen2.5-3B-Instruct (3B parameters, transformer, causal LM)
- **Adapter:** LoRA, r=16, α=32; ~67M trainable params (0.4% of base)
- **Precision:** bf16 (merged); quantized to q8_0, q4_k_m for deployment
- **Minimum inference (CPU, q4_k_m):** Apple Silicon (M1+), x86-64 (Ryzen 5000+, Intel 12th gen+)
- **Ollama:** Runs on macOS, Linux, Windows

## Citation

```bibtex
@misc{spanfinder-3b,
  title={praxis/spanfinder-3b: Lightweight PII Span Detection for On-Device Disclosure Control},
  author={Praxis},
  year={2026},
  url={https://huggingface.co/praxis-nation/spanfinder-3b},
  note={Fine-tuned Qwen2.5-3B-Instruct LoRA. Base model license: Qwen Research License (non-commercial).}
}
```

## Glossary

- **LoRA:** Low-Rank Adaptation; parameter-efficient fine-tuning via small trainable adapters added to a frozen base model
- **QLoRA:** LoRA with 4-bit quantization; reduces VRAM during training
- **GGUF:** Quantized model format optimized for CPU inference (llama.cpp, Ollama)
- **PII:** Personally Identifiable Information (names, locations, organizations, emails, phone numbers)
- **Span:** A contiguous substring of text (e.g., "Toronto" or "Microsoft")
- **Disclosure control:** Reducing PII exposure in transmitted data; distinct from a privacy guarantee

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
