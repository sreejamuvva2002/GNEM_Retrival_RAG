# Quick Start: Qwen Fine-Tuning Pipeline

This guide walks you through running the complete fine-tuning pipeline to generate 500-1000+ synthetic training examples and prepare them for fine-tuning.

## Prerequisites

✅ **Already have:**
- 50 human-validated Q&A pairs (`kb/Human validated 50 questions.xlsx`)
- Knowledge Base (`kb/GNEM - Auto Landscape Lat Long Updated.xlsx`)
- Ollama running with Qwen2.5:14b

⚠️ **For GPT-4o/Gemini (optional, for better data quality):**
- OpenAI API key: `OPENAI_API_KEY=sk-...`
- OR Google Gemini API key: `GOOGLE_API_KEY=...`

## Option 1: Using Local Ollama with Large Model (Free, Best Quality)

**Recommended:** Use **Llama 2 70B** (or Mistral 7B if you have less VRAM)

### Setup (5-30 minutes depending on internet)

1. **Pull the large model:**
   ```bash
   # Option A: Llama 2 70B (best quality, requires 45GB VRAM)
   ollama pull llama2:70b
   
   # Option B: Mistral 7B (good quality, requires 8GB VRAM, faster)
   ollama pull mistral
   
   # Option C: Orca 2 13B (good balance, requires 10GB VRAM)
   ollama pull orca-mini:13b
   ```

2. **Verify Ollama is running:**
   ```bash
   curl http://localhost:11434/api/tags
   # Should show your pulled model
   ```

3. **Add to `.env`:**
   ```bash
   DATA_GEN_LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   DATA_GEN_OLLAMA_MODEL=llama2:70b    # or mistral / orca-mini:13b
   DATA_GEN_TEMPERATURE=0.7
   ```

4. **Run the pipeline:**
   ```bash
   python -m georgia_ev_intelligence.finetuning.cli pipeline
   ```

   This will:
   - Generate ~800-1500 synthetic Q&A pairs (paraphrases + KB-driven)
   - Validate each pair (LLM-as-judge)
   - Format into `train_dataset.jsonl` and `val_dataset.jsonl`

   **Expected runtime:**
   - Llama 2 70B: 3-6 hours (best quality)
   - Mistral 7B: 1-2 hours (good quality, faster)
   - Orca 2 13B: 2-3 hours (excellent quality)

**For detailed setup help:** See [OLLAMA_SETUP_GUIDE.md](OLLAMA_SETUP_GUIDE.md)

---

## Option 2: Using GPT-4o (Fastest, Very High Quality)

### Setup (2 minutes)

1. **Get OpenAI API key:**
   - Visit https://platform.openai.com/api/keys
   - Create/copy your API key

2. **Add to `.env`:**
   ```bash
   DATA_GEN_LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-<your-key>
   DATA_GEN_OPENAI_MODEL=gpt-4o
   DATA_GEN_TEMPERATURE=0.7
   DATA_GEN_MAX_TOKENS=2048
   ```

3. **Run the pipeline:**
   ```bash
   python -m georgia_ev_intelligence.finetuning.cli pipeline \
     --paraphrase-count 10 \
     --kb-questions-per-chunk 5
   ```

   **Expected runtime:** 30 minutes to 1 hour  
   **Cost:** ~$5-15 USD (for ~1500 examples)

---

## Option 3: Using vLLM with Local Large Model (GPU Alternative)

### Setup (15 minutes)

1. **Start vLLM server with 120B model:**
   ```bash
   # Install vLLM if needed
   pip install vllm

   # Start server (requires ~80GB VRAM for 120B model)
   python -m vllm.entrypoints.openai.api_server \
     --model meta-llama/Llama-2-70b-hf \
     --dtype float16 \
     --gpu-memory-utilization 0.9
   ```

   Or for smaller model:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model mistralai/Mistral-7B-Instruct-v0.2 \
     --gpu-memory-utilization 0.9
   ```

2. **Add to `.env`:**
   ```bash
   DATA_GEN_LLM_PROVIDER=vllm
   VLLM_BASE_URL=http://localhost:8000
   DATA_GEN_VLLM_MODEL=meta-llama/Llama-2-70b-hf
   DATA_GEN_TEMPERATURE=0.7
   ```

3. **Run the pipeline:**
   ```bash
   python -m georgia_ev_intelligence.finetuning.cli pipeline
   ```

---

## Step-by-Step Execution (If You Want More Control)

### Step 1: Generate Synthetic Q&A Pairs

```bash
python -m georgia_ev_intelligence.finetuning.cli augment \
  --paraphrase-count 8 \
  --kb-questions-per-chunk 3
```

**Output:** `georgia_ev_intelligence/outputs/finetuning/augmented_questions.jsonl`  
**Expected:** ~1000-1500 Q&A pairs

### Step 2: Validate with LLM-as-Judge

```bash
python -m georgia_ev_intelligence.finetuning.cli validate \
  --min-score 4
```

**Output:** 
- `validated_questions.jsonl` (high-quality pairs only)
- `validation_report.json` (scoring breakdown)

**Expected:** ~500-1000 pairs pass validation

### Step 3: Format for Fine-Tuning

```bash
python -m georgia_ev_intelligence.finetuning.cli format \
  --train-ratio 0.8
```

**Outputs:**
- `train_dataset.jsonl` (80% of pairs)
- `val_dataset.jsonl` (20% of pairs)

### Step 4: Review the Dataset

```python
import json

# Check training data
with open("georgia_ev_intelligence/outputs/finetuning/train_dataset.jsonl") as f:
    examples = [json.loads(line) for line in f]
    print(f"Training examples: {len(examples)}")
    print(f"First example:\n{json.dumps(examples[0], indent=2)}")
```

---

## Configuration Tuning

### For More Synthetic Data (Lower Quality)
```bash
python -m georgia_ev_intelligence.finetuning.cli pipeline \
  --paraphrase-count 15 \
  --kb-questions-per-chunk 5 \
  --min-score 3
```

### For Higher Quality (Fewer Examples)
```bash
python -m georgia_ev_intelligence.finetuning.cli pipeline \
  --paraphrase-count 5 \
  --kb-questions-per-chunk 2 \
  --min-score 5
```

### Recommended for Your Setup
```bash
# Using local Qwen2.5 (free, ~2-3 hours)
python -m georgia_ev_intelligence.finetuning.cli pipeline \
  --paraphrase-count 8 \
  --kb-questions-per-chunk 3

# Using GPT-4o (faster, ~30 min, ~$10 cost)
python -m georgia_ev_intelligence.finetuning.cli pipeline \
  --paraphrase-count 10 \
  --kb-questions-per-chunk 5
```

---

## What to Do Next

### After data is prepared:

1. **Fine-tune the model** (Phase 4):
   ```bash
   # Coming in next phase
   python -m georgia_ev_intelligence.finetuning.qwen_finetuner \
     --model Qwen/Qwen2.5-14B \
     --train-data train_dataset.jsonl \
     --val-data val_dataset.jsonl \
     --output-dir checkpoints/qwen_finetuned
   ```

2. **Evaluate** (Phase 5):
   ```bash
   # Coming in next phase
   python -m georgia_ev_intelligence.finetuning.evaluation \
     --model-path checkpoints/qwen_finetuned
   ```

3. **Deploy to Ollama** (Phase 6):
   ```bash
   # Coming in next phase
   python -m georgia_ev_intelligence.finetuning.ollama_integration \
     --model-path checkpoints/qwen_finetuned
   ```

---

## Troubleshooting

### "Connection refused" to Ollama
```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify it's working
curl http://localhost:11434/api/tags
```

### "OpenAI API key not found"
```bash
# Make sure .env has:
OPENAI_API_KEY=sk-...
```

### "JSON decode error"
- The LLM is having trouble returning valid JSON
- Try:
  1. Increase `DATA_GEN_TEMPERATURE=0.5` (lower is more consistent)
  2. Use a different/better LLM
  3. Check LLM logs for errors

### Low validation acceptance rate (< 30%)
- Increase `VALIDATION_MIN_SCORE` lower (try 3)
- Use a stronger LLM for data generation (GPT-4o)
- Review `validation_report.json` to see what's failing

---

## Example Output

After completing the pipeline, you should have:

```
georgia_ev_intelligence/outputs/finetuning/
├── augmented_questions.jsonl      # ~1500 raw pairs
├── validated_questions.jsonl      # ~1000 high-quality pairs
├── validation_report.json         # Scoring stats
├── train_dataset.jsonl            # ~800 training pairs (ChatML format)
├── val_dataset.jsonl              # ~200 validation pairs (ChatML format)
└── checkpoints/
    └── (fine-tuned models go here in Phase 4)
```

**Sample record from `train_dataset.jsonl`:**
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert assistant specialized in Georgia EV Intelligence..."
    },
    {
      "role": "user",
      "content": "How long does it take to charge a Tesla at Level 2?"
    },
    {
      "role": "assistant",
      "content": "Level 2 charging typically takes 4-10 hours depending on battery size and charger power..."
    }
  ]
}
```

---

## Performance Expectations

| Step | Local Ollama | GPT-4o | vLLM 70B |
|------|--------------|--------|----------|
| Augmentation | 1-2 hours | 5-10 min | 15-30 min |
| Validation | 1-2 hours | 15-30 min | 30-60 min |
| Formatting | < 1 min | < 1 min | < 1 min |
| **Total** | **2-4 hours** | **30 min** | **1 hour** |
| **Cost** | **Free** | **~$10** | **Free** |

---

## Need Help?

1. Check `georgia_ev_intelligence/finetuning/README.md` for detailed documentation
2. Review logs for specific errors
3. See `FINETUNING_PLAN.md` for the full strategy
