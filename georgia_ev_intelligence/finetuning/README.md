# Fine-Tuning Module for Qwen Model

This module implements a complete fine-tuning pipeline for the Qwen model using Knowledge Distillation and QLoRA. It expands a small set of validated Q&A pairs into a large synthetic dataset and fine-tunes a local Qwen model to become an EV Intelligence expert.

## Overview

The pipeline has three main phases:

### Phase 1: Data Augmentation
- **Paraphrasing:** Generate variations of validated questions using a powerful LLM
- **KB-Driven Generation:** Generate entirely new Q&A pairs grounded in your Knowledge Base
- **Adversarial:** Generate "I don't know" questions to improve robustness

### Phase 2: Validation
- **LLM-as-Judge:** Score each generated pair on accuracy, completeness, and relevance
- **Quality Filtering:** Discard low-quality pairs automatically
- **Deduplication:** Remove semantic near-duplicates

### Phase 3: Formatting & Fine-Tuning
- **ChatML Format:** Convert to Qwen-compatible JSONL format
- **Train/Val Split:** 80/20 split with stratification
- **Fine-Tuning:** QLoRA-based training with Unsloth (Phase 4, implemented separately)

## Setup

### 1. Configure LLM Provider

Choose where to run the synthetic data generation. Options:

#### Option A: Local Model via Ollama (Recommended)
```bash
# Set in .env
DATA_GEN_LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DATA_GEN_OLLAMA_MODEL=qwen2.5:14b  # or your preferred model
```

#### Option B: Local Model via vLLM
```bash
# Start vLLM server with 120B model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-hf \
  --dtype float16 \
  --gpu-memory-utilization 0.9

# Set in .env
DATA_GEN_LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8000
DATA_GEN_VLLM_MODEL=meta-llama/Llama-2-70b
```

#### Option C: OpenAI API
```bash
# Set in .env
DATA_GEN_LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
DATA_GEN_OPENAI_MODEL=gpt-4o
```

#### Option D: Google Gemini API
```bash
# Set in .env
DATA_GEN_LLM_PROVIDER=gemini
GOOGLE_API_KEY=...
DATA_GEN_GEMINI_MODEL=gemini-1.5-pro
```

### 2. Hyperparameter Tuning

Edit `.env` to customize:

```bash
# Data augmentation
PARAPHRASE_COUNT=8                    # Variations per question
KB_QUESTIONS_PER_CHUNK=3              # Questions per KB chunk
INCLUDE_ADVERSARIAL=true              # Generate "I don't know" questions

# Validation
VALIDATION_MIN_SCORE=4                # Min score (1-5) to accept
DEDUP_SIMILARITY_THRESHOLD=0.85       # Semantic dedup threshold

# Fine-tuning
LORA_RANK=32
LORA_ALPHA=64
TRAINING_EPOCHS=2
LEARNING_RATE=2e-4
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
MAX_SEQ_LENGTH=2048
```

## Usage

### Quick Start: Run Full Pipeline

```bash
python -m georgia_ev_intelligence.finetuning.cli pipeline
```

This runs all three phases in sequence:
1. Generates synthetic Q&A pairs
2. Validates them (LLM-as-judge)
3. Formats into ChatML JSONL for fine-tuning

### Step-by-Step Execution

#### Step 1: Generate Synthetic Q&A Pairs
```bash
python -m georgia_ev_intelligence.finetuning.cli augment \
  --paraphrase-count 8 \
  --kb-questions-per-chunk 3
```

**Output:** `georgia_ev_intelligence/outputs/finetuning/augmented_questions.jsonl`

#### Step 2: Validate with LLM-as-Judge
```bash
python -m georgia_ev_intelligence.finetuning.cli validate \
  --input georgia_ev_intelligence/outputs/finetuning/augmented_questions.jsonl \
  --min-score 4
```

**Outputs:**
- `validated_questions.jsonl` - High-quality pairs only
- `validation_report.json` - Detailed scoring breakdown

#### Step 3: Format for Fine-Tuning
```bash
python -m georgia_ev_intelligence.finetuning.cli format \
  --input georgia_ev_intelligence/outputs/finetuning/validated_questions.jsonl \
  --train-ratio 0.8
```

**Outputs:**
- `train_dataset.jsonl` - 80% of validated pairs
- `val_dataset.jsonl` - 20% of validated pairs

### Using as a Python Module

```python
from georgia_ev_intelligence.finetuning.data_augmentation import augment_dataset
from georgia_ev_intelligence.finetuning.validation import validate_augmented_dataset
from georgia_ev_intelligence.finetuning.dataset_formatter import format_validated_dataset

# Step 1: Augment
augmented = augment_dataset(
    paraphrase_count=8,
    kb_questions_per_chunk=3,
    include_adversarial=True
)

# Step 2: Validate
passed, failed = validate_augmented_dataset(min_score=4)
print(f"Accepted: {len(passed)}, Rejected: {len(failed)}")

# Step 3: Format
train_path, val_path = format_validated_dataset(train_ratio=0.8)
```

## Module Architecture

```
georgia_ev_intelligence/finetuning/
├── config.py                 # Configuration & hyperparameters
├── llm_client.py            # LLM abstraction (Ollama, vLLM, OpenAI, Gemini)
├── data_augmentation.py     # Phase 1: Generate synthetic Q&A
├── validation.py            # Phase 2: LLM-as-judge scoring
├── dataset_formatter.py     # Phase 3: ChatML formatting
├── deduplication.py         # Semantic deduplication (planned)
├── coverage_analyzer.py     # Topic coverage analysis (planned)
├── qwen_finetuner.py       # Phase 4: Fine-tuning with Unsloth (planned)
├── evaluation.py            # Phase 5: Evaluation metrics (planned)
├── ollama_integration.py   # Phase 6: Ollama deployment (planned)
├── cli.py                  # Command-line interface
└── README.md              # This file
```

## Data Flow

```
Human validated 50 questions.xlsx
    ↓
[Phase 1] Data Augmentation
├─→ Paraphraser (8 variations per question)
├─→ KB Question Generator (3 questions per KB chunk)
└─→ Adversarial Generator (50 "I don't know" questions)
    ↓
augmented_questions.jsonl (~1500-2000 pairs)
    ↓
[Phase 2] Validation (LLM-as-Judge)
├─→ Score accuracy, completeness, relevance
├─→ Filter (min_score ≥ 4)
└─→ Deduplicate (cosine sim threshold)
    ↓
validated_questions.jsonl (~1000-1500 pairs)
    ↓
[Phase 3] Dataset Formatting
├─→ Convert to ChatML format
└─→ Train/Validation split (80/20)
    ↓
train_dataset.jsonl + val_dataset.jsonl
    ↓
[Phase 4] Fine-Tuning (via qwen_finetuner.py)
├─→ QLoRA with Unsloth
└─→ 2 epochs, LR=2e-4
    ↓
checkpoints/qwen_finetuned/adapter_model.bin
```

## Configuration Reference

### LLM Provider Selection

| Provider | Setup | Cost | Speed | Quality |
|----------|-------|------|-------|---------|
| **Ollama** | Local | Free | Medium | Depends on model |
| **vLLM** | Local (GPU) | Free | Fast | High |
| **OpenAI** | API key | $$$ | Fast | Excellent |
| **Gemini** | API key | $$ | Fast | Excellent |

### Recommended Configurations

**For Free Local Fine-Tuning:**
- Use Ollama with Qwen2.5:14b (already in your setup)
- 8 paraphrases per question
- 3 KB questions per chunk
- Min validation score: 4

**For Maximum Quality:**
- Use OpenAI GPT-4o or Google Gemini
- 10+ paraphrases per question
- 5+ KB questions per chunk
- Min validation score: 4.5

## Output Files

| File | Format | Purpose |
|------|--------|---------|
| `augmented_questions.jsonl` | JSONL | All generated Q&A pairs (before filtering) |
| `validated_questions.jsonl` | JSONL | High-quality pairs only (after validation) |
| `validation_report.json` | JSON | Detailed scoring breakdown & stats |
| `train_dataset.jsonl` | JSONL | Training set in ChatML format |
| `val_dataset.jsonl` | JSONL | Validation set in ChatML format |

Each JSONL file has one JSON object per line:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## Troubleshooting

### "LLM request timeout"
- Increase `DATA_GEN_TIMEOUT` in `.env` (default 300s)
- Check if Ollama/vLLM server is running
- Try a smaller model

### "Low acceptance rate" (< 50% validated)
- Adjust `VALIDATION_MIN_SCORE` lower (try 3 instead of 4)
- Use a more capable LLM for data generation
- Review validation_report.json to see what's failing

### "Out of memory"
- Reduce `PARAPHRASE_COUNT`
- Reduce `KB_QUESTIONS_PER_CHUNK`
- Process smaller KB chunks

### "JSON decode errors"
- Check if LLM is properly returning JSON
- Increase `DATA_GEN_MAX_TOKENS` if responses are truncated
- Try a different LLM provider

## Next Steps

After completing the data preparation, proceed to:

1. **Fine-tuning** (Phase 4): Use `georgia_ev_intelligence.finetuning.qwen_finetuner`
2. **Evaluation** (Phase 5): Compare fine-tuned vs. base model
3. **Deployment** (Phase 6): Deploy to Ollama or other inference engine

See `FINETUNING_PLAN.md` for the full roadmap.

## References

- [Qwen Model](https://github.com/QwenLM/Qwen)
- [Unsloth](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Knowledge Distillation](https://arxiv.org/abs/1503.02531)
