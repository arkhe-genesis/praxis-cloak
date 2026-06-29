# Licensing and Attribution Notice

This file covers the licensing and attribution requirements for the model artifacts in this repository.

---

## Base Model License: Qwen Research License (Non-Commercial)

Both **praxis/spanfinder-3b** and **praxis/relevance-3b** are fine-tuned from **Qwen/Qwen2.5-3B-Instruct**, which is distributed under the **Qwen Research License Agreement**.

**Required notice (must appear in all distributions of these model weights):**

> Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.

Full license text: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE

### Key Terms

- **Permitted use:** Research or evaluation purposes only (non-commercial).
- **Commercial use:** Requires a separate, written license from Alibaba Cloud (https://www.aliyun.com/). These model weights may **not** be used commercially without that license.
- **Attribution:** Any redistribution must retain the copyright notice above.
- **Derivative works:** Distributions of fine-tuned or modified versions must display **"Built with Qwen"** or **"Improved using Qwen"** prominently in product documentation.
- **Warranty:** Materials are provided "AS IS" with no warranty. Alibaba Cloud retains all intellectual property rights to the original Qwen model.

### Our Compliance

All published artifacts (model cards, Modelfiles, NOTICE, README) display "Improved using Qwen" and include the required Alibaba Cloud copyright notice. We link to the original Qwen Research License in every distribution point.

---

## Training Data: CAPID Dataset (Relevance Model Only)

**Applies to:** praxis/relevance-3b

**Dataset:** CAPID — Context-Aware PII Detection
**License:** MIT
**Source:** https://github.com/MariaPonomarenko38/CAPID

The relevance-3b model was trained on a subset of the CAPID dataset (1,897 training examples from CAPID's real Reddit relocation/advice posts, plus Praxis-generated synthetic data). CAPID's test and reddit splits were held out for evaluation and were never trained on.

MIT license requires attribution only:

> CAPID is MIT-licensed. Copyright (c) MariaPonomarenko38 and contributors.
> https://github.com/MariaPonomarenko38/CAPID

No additional restrictions apply beyond attribution.

---

## Training Data: Spanfinder Private Dataset (Spanfinder Model Only)

**Applies to:** praxis/spanfinder-3b

The spanfinder model was trained on real interview QA turns from the private Praxis dataset. This data contains real PII and is **NOT published**. Only the trained model weights are distributed.

- No `.jsonl` training files are or will be included in this repository or any HuggingFace release.
- The weights themselves do not contain or reproduce raw training data.

---

## Repo Code License

The source code in this repository (everything outside of `models/*.gguf` and `models/*.safetensors`) is MIT-licensed. See the root `LICENSE` file.

**Model artifacts** (GGUF files, safetensors, LoRA adapters) are governed by the Qwen Research License (non-commercial), not MIT.

---

## Summary Table

| Artifact | License | Commercial Use |
|---|---|---|
| Repo source code | MIT | Yes |
| praxis/spanfinder-3b weights | Qwen Research License | No — requires separate Alibaba Cloud license |
| praxis/relevance-3b weights | Qwen Research License | No — requires separate Alibaba Cloud license |
| CAPID dataset (relevance training) | MIT | Yes (attribution required) |
| Spanfinder training data | Private (contains real PII) | Not published |
