# Ollama Setup Guide: Running Large Models for Data Generation

This guide walks you through setting up Ollama with a large open-source model (70B+) for high-quality synthetic data generation.

## Overview

For fine-tuning data generation, using a large model produces better quality than smaller ones. We recommend **Llama 2 70B** or similar.

| Model | Size | VRAM | Quality | Speed |
|-------|------|------|---------|-------|
| Qwen2.5:14B | 14B | 9GB | Good | Fast |
| **Llama 2:70B** | 70B | 45GB | Excellent | Slow |
| Mistral | 7B | 5GB | Good | Fast |
| Neural Chat | 7B | 5GB | Good | Fast |

**Recommendation:** Use **Llama 2 70B** for best synthetic data quality (if you have 45GB+ VRAM)

---

## Prerequisites

### Hardware Requirements

**For Llama 2 70B:**
- GPU with 45-50GB VRAM minimum (RTX 6000, A100, H100, etc.)
- OR 2+ GPUs with 24GB each (RTX 4090, RTX 3090, etc.)

**For Mistral 7B / Neural Chat (faster, lower quality):**
- GPU with 8GB+ VRAM (RTX 3060, RTX 4060, etc.)

**CPU fallback:**
- If no GPU: Use smaller model (7B), will be very slow (~1 token/sec)

### Software Requirements

1. **Ollama** installed and running
   - Download from https://ollama.ai
   - Or: `brew install ollama` (macOS), `snap install ollama` (Linux)

2. **Python packages** (already have from your .env setup)

---

## Step 1: Install Ollama

### macOS / Linux
```bash
curl https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve
# Server runs at http://localhost:11434
```

### Windows
Download installer from https://ollama.ai/download

Then start the server:
```bash
ollama serve
```

### Docker
```bash
docker run -it --gpus=all -v ollama:/root/.ollama -p 11434:11434 ollama/ollama

# In another terminal
docker exec <container-id> ollama pull llama2:70b
```

---

## Step 2: Pull Your Large Model

### Option A: Llama 2 70B (Recommended, Highest Quality)

```bash
# This downloads ~45GB (takes 20-30 minutes depending on internet)
ollama pull llama2:70b

# Verify it's installed
ollama list
# Should show: llama2:70b   40GB
```

**Llama 2 70B Best For:**
- ✅ Highest quality synthetic data
- ✅ Best reasoning and domain knowledge
- ✅ Best paraphrases and variations
- ❌ Slower (takes 1-3 min per request)
- ❌ Requires 45GB+ VRAM

### Option B: Mistral (Good quality, Faster)

```bash
ollama pull mistral
# ~5GB, much faster

# Or pull a specific version
ollama pull mistral:7b-instruct
```

**Mistral Best For:**
- ✅ Good quality, much faster (~30s per request)
- ✅ Requires only 8GB VRAM
- ❌ Slightly lower quality than 70B
- ❌ Less diverse outputs

### Option C: Other Large Models

```bash
# Orca 2 (good reasoning, 13B)
ollama pull orca-mini:13b

# Neural Chat (good conversational, 7B)
ollama pull neural-chat

# Dolphin Mixtral (great quality, 7B)
ollama pull dolphin-mixtral:8x7b

# Starling (good, 7B)
ollama pull starling-lm
```

---

## Step 3: Test Your Model

```bash
# Test if Ollama is running
curl http://localhost:11434/api/tags

# Should return something like:
# {
#   "models": [
#     {"name": "llama2:70b", "modified_at": "2024-06-30..."}
#   ]
# }

# Test generating from your model
curl http://localhost:11434/api/generate -d '{
  "model": "llama2:70b",
  "prompt": "What is machine learning?",
  "stream": false
}'
```

---

## Step 4: Configure for Fine-Tuning

### 1. Create/Update `.env`:

```bash
# Copy from template
cp .env.finetuning.template .env

# Edit to use your large model
# DATA_GEN_LLM_PROVIDER=ollama
# DATA_GEN_OLLAMA_MODEL=llama2:70b
```

### 2. Verify Configuration:

```bash
python -c "
from georgia_ev_intelligence.finetuning import config
print(f'Provider: {config.DATA_GEN_LLM_PROVIDER}')
print(f'Model: {config.DATA_GEN_OLLAMA_MODEL}')
print(f'Base URL: {config.OLLAMA_BASE_URL}')
"
```

---

## Step 5: Run Data Augmentation

### Test with Small Dataset First:

```bash
python -m georgia_ev_intelligence.finetuning.cli augment \
  --paraphrase-count 2 \
  --kb-questions-per-chunk 1
```

**Expected output after ~5-10 minutes:**
- Few log messages
- File: `georgia_ev_intelligence/outputs/finetuning/augmented_questions.jsonl`
- Contains ~100 Q&A pairs (2 paraphrases × 50 questions + 50 KB questions)

### Full Dataset Generation:

```bash
python -m georgia_ev_intelligence.finetuning.cli pipeline \
  --paraphrase-count 8 \
  --kb-questions-per-chunk 3
```

**Expected runtime:** 
- Llama 2 70B: 3-6 hours
- Mistral 7B: 1-2 hours
- Orca 2: 2-3 hours

**Output files:**
- `augmented_questions.jsonl` (~1500 Q&A pairs)
- `validated_questions.jsonl` (~1000 after filtering)
- `train_dataset.jsonl` (~800 training examples)
- `val_dataset.jsonl` (~200 validation examples)

---

## GPU Memory Optimization

### If You Run Out of Memory

**Option 1: Use Smaller Model**
```bash
ollama pull mistral  # 7B instead of 70B
# Update .env: DATA_GEN_OLLAMA_MODEL=mistral
```

**Option 2: Use Quantized Version**
```bash
# Ollama automatically uses quantization for large models
# But you can try smaller variants
ollama pull llama2:13b  # 13B version (16GB VRAM)
```

**Option 3: Enable CPU Offloading**
```bash
# Ollama settings (if supported by your GPU)
# Usually handled automatically, but you can configure in ~/.ollama/ollama.config
```

**Option 4: Batch Processing**
```bash
# Process smaller batches, save results
python -m georgia_ev_intelligence.finetuning.cli augment \
  --paraphrase-count 4  # Reduce batch size
```

---

## Troubleshooting

### "Connection refused" to Ollama

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify
curl http://localhost:11434/api/tags

# If still failing, check port
lsof -i :11434
```

### "Model not found"

```bash
# List installed models
ollama list

# Pull the model
ollama pull llama2:70b

# If pull fails, check disk space
df -h
```

### Very Slow Generation (1+ minute per request)

- **Llama 2 70B is naturally slow** - this is expected
- If slower than expected:
  - Check GPU utilization: `nvidia-smi` (should show 90%+ GPU usage)
  - Try a smaller model: `ollama pull mistral`
  - Check if other processes are using GPU

### Out of Memory (OOM) Errors

```bash
# Check GPU memory
nvidia-smi

# Solutions:
# 1. Use smaller model (mistral instead of llama2:70b)
# 2. Reduce batch size (--paraphrase-count 4 instead of 8)
# 3. Enable CPU offloading if available
# 4. Use multiple smaller requests instead of one large batch
```

### JSON Decode Errors

- Model is not returning valid JSON
- Happens occasionally with generation
- Solutions:
  1. Increase `DATA_GEN_TEMPERATURE=0.5` (lower = more consistent)
  2. Use a different model
  3. Try again (some responses will fail, pipeline continues)

---

## Performance Comparison

Running the full augmentation pipeline (50 original questions → synthetic pairs):

| Model | Time | VRAM | Quality | Cost |
|-------|------|------|---------|------|
| Llama 2 70B | 4-6 hours | 45GB | ⭐⭐⭐⭐⭐ | Free |
| Mistral 7B | 1-2 hours | 8GB | ⭐⭐⭐⭐ | Free |
| Neural Chat 7B | 1-2 hours | 8GB | ⭐⭐⭐⭐ | Free |
| Orca 2 13B | 2-3 hours | 10GB | ⭐⭐⭐⭐⭐ | Free |
| GPT-4o (API) | 30 min | Cloud | ⭐⭐⭐⭐⭐ | $10-15 |

---

## Running Multiple Models

You can run multiple Ollama instances on different ports:

```bash
# Terminal 1: Llama 2 70B on default port
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Terminal 2: Mistral on different port
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

Then in `.env`:
```bash
# Use the first for data generation
DATA_GEN_OLLAMA_MODEL=llama2:70b
OLLAMA_BASE_URL=http://localhost:11434

# Use the second for production inference
OLLAMA_LLM_MODEL=mistral
# (from existing runtime config)
```

---

## Recommended Setup

### Best Quality (for fine-tuning)
```bash
# Use Llama 2 70B for data generation
DATA_GEN_OLLAMA_MODEL=llama2:70b
DATA_GEN_TEMPERATURE=0.7

# Use faster model for runtime inference
OLLAMA_LLM_MODEL=mistral  # For actual RAG queries
```

### Balanced (Good quality, reasonable speed)
```bash
# Orca 2 13B is a sweet spot
DATA_GEN_OLLAMA_MODEL=orca-mini:13b
DATA_GEN_TEMPERATURE=0.7
```

### Fast (Lower quality but quick)
```bash
# Mistral 7B is fast and good
DATA_GEN_OLLAMA_MODEL=mistral
DATA_GEN_TEMPERATURE=0.5  # Lower for consistency
```

---

## Advanced: Pull Custom Models

If you have a custom/fine-tuned model:

```bash
# From Hugging Face
ollama pull hf.co/username/model-name

# From GGUF file locally
ollama create mymodel -f Modelfile
# Where Modelfile contains:
# FROM ./model.gguf
# PARAMETER temperature 0.7
```

---

## References

- **Ollama GitHub:** https://github.com/ollama/ollama
- **Available Models:** https://ollama.ai/library
- **Ollama Documentation:** https://github.com/ollama/ollama/tree/main/docs
- **Llama 2:** https://huggingface.co/meta-llama/Llama-2-70b
- **Mistral:** https://huggingface.co/mistralai/Mistral-7B-v0.1

---

## Quick Reference

```bash
# Install
curl https://ollama.ai/install.sh | sh

# Start server
ollama serve

# List models
ollama list

# Pull a model
ollama pull llama2:70b  # 45GB
ollama pull mistral     # 5GB
ollama pull orca-mini:13b  # 8GB

# Remove a model
ollama rm llama2:70b

# Test model
curl http://localhost:11434/api/tags

# Run fine-tuning pipeline
python -m georgia_ev_intelligence.finetuning.cli pipeline
```
