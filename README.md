# Cloak by Praxis

Chat with cloud LLMs while a local model swaps your PII out before the prompt leaves your machine — and rehydrates the answer back to your real entities.

## The idea

Every time you ask a cloud AI about something personal — a city you're moving to, a company you work for, a name — you're sending that data to an external server. Cloak intercepts each prompt, runs a local PII detector, replaces sensitive entities with realistic stand-ins (not `[REDACTED]` — actual fake names and cities), sends the scrubbed prompt to the cloud, then maps the answer back to your real entities before you see it.

The round-trip substitution is the differentiator: the cloud model sees a coherent, answerable prompt; you get a useful, real answer; and the actual PII never left your device.

This is **disclosure control**, not a privacy guarantee. The pipeline reduces PII exposure significantly — but a determined adversary with enough queries could still reconstruct context. The goal is to make routine LLM use substantially less leaky, without sacrificing answer quality.

## Quickstart

**Prerequisites:** [Ollama](https://ollama.com/download) installed and running.

```bash
# 1. Install Ollama (if not already):
#    https://ollama.com/download

# 2. Clone and launch:
git clone https://github.com/praxis-society/praxis-cloak.git
cd praxis-cloak
./scripts/run.sh
```

`run.sh` will:
- Download `praxis/spanfinder-3b` and `praxis/relevance-3b` from HuggingFace and build them into Ollama locally (first run only, ~3.8 GB total — no Ollama account needed)
- Create a Python virtual environment and install dependencies
- Build the frontend (requires Node; skipped if `npm` is not found)
- Start the local server at `http://127.0.0.1:8765` and open it in your browser

**Settings:** Paste your Anthropic or OpenAI API key in Settings to route scrubbed prompts through your preferred provider.

## Models

| Model | Weights | Size (q4_k_m) | Role |
|---|---|---|---|
| `praxis/spanfinder-3b` | [HuggingFace](https://huggingface.co/praxis-nation/spanfinder-3b) | ~1.8 GB | Detects PII spans in the prompt |
| `praxis/relevance-3b` | [HuggingFace](https://huggingface.co/praxis-nation/relevance-3b) | ~1.8 GB | Decides which detected entities to scrub vs. keep (needed to answer the question) |

`run.sh` downloads these GGUFs from HuggingFace and builds them into Ollama for you (the relevance model ships a chat template via its Modelfile, so build it locally rather than pulling the bare GGUF). Both are Qwen2.5-3B-Instruct LoRA fine-tunes; q8_0 reference quantizations (~3.1 GB each) and the LoRA adapters are also on HuggingFace. To try the span detector standalone: `ollama run hf.co/praxis-nation/spanfinder-3b:Q4_K_M`.

## How it works

```
User prompt
    │
    ▼
[regex fast-pass]          ← catches obvious patterns (email, phone)
    │
    ▼
[spanfinder-3b]            ← on-device; detects PII spans (names, orgs, locations)
    │
    ▼
[relevance-3b]             ← decides: KEEP (load-bearing) or SCRUB (incidental)
    │
    ▼
[substitution]             ← replaces SCRUB entities with realistic fakes
    │                         (not [REDACTED]; actual coherent stand-ins)
    ▼
[cloud LLM]                ← sees a coherent, answerable, scrubbed prompt
    │
    ▼
[rehydration]              ← maps fake entities in the answer back to real ones
    │
    ▼
Answer (real entities)
```

**Re relevance step -** Keyword scrubbing over-removes: "What's the tax rate in Toronto?" needs "Toronto" to produce a useful answer. The relevance model judges whether each detected entity is *needed* — and only the incidental ones get substituted.

**Re realistic fakes -** `[REDACTED]` breaks the cloud model's ability to reason coherently. A fake city name preserves sentence structure and answer quality; the rehydration step maps it back.

## Repo layout

```
app/
  backend/          FastAPI server (scrub pipeline, cloud calls, rehydration)
  frontend/         React UI
models/
  spanfinder-3b/    HuggingFace model card
  relevance-3b/     HuggingFace model card
  Modelfile.spanfinder
  Modelfile.relevance
  LICENSE_AND_ATTRIBUTION.md
scripts/
  run.sh            One-command launcher
  install.sh        Dependency installer (no server start)
src/                Core library (span detection, substitution, pipeline)
```

## License

**Source code:** MIT. Copyright (c) 2026 Praxis. See `LICENSE`.

**Model artifacts** (`*.gguf`, `*.safetensors`, LoRA adapter weights): Qwen Research License (non-commercial). Commercial use requires a separate license from Alibaba Cloud.

> Built with Qwen — praxis/spanfinder-3b and praxis/relevance-3b are fine-tuned derivatives of Qwen/Qwen2.5-3B-Instruct.
> *Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.*

See `NOTICE` and `models/LICENSE_AND_ATTRIBUTION.md` for full attribution including the CAPID dataset (MIT) used to train the relevance model.
