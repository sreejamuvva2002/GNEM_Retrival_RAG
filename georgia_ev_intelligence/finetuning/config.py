"""Configuration for fine-tuning pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = ROOT / "georgia_ev_intelligence"
KB_DIR = ROOT / "kb"
OUTPUTS_DIR = PACKAGE_DIR / "outputs"
FINETUNING_OUTPUTS_DIR = OUTPUTS_DIR / "finetuning"

# Create output directory if it doesn't exist
FINETUNING_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(ROOT / ".env")


def _env_optional_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_optional_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


# ============================================================================
# LLM Configuration for Data Augmentation & Validation
# ============================================================================

# Which LLM to use for data generation: "ollama", "vllm", "openai", "gemini"
DATA_GEN_LLM_PROVIDER = _env_optional_str("DATA_GEN_LLM_PROVIDER", "ollama")

# ---- Ollama Configuration (Local) ----
# For data generation, use a large model: llama2:70b, mistral, neural-chat, etc.
# Recommended: llama2:70b for best quality (requires ~45GB VRAM)
OLLAMA_BASE_URL = _env_optional_str("OLLAMA_BASE_URL", "http://localhost:11434")
DATA_GEN_OLLAMA_MODEL = _env_optional_str("DATA_GEN_OLLAMA_MODEL", "llama2:70b")

# ---- vLLM Configuration (Local) ----
VLLM_BASE_URL = _env_optional_str("VLLM_BASE_URL", "http://localhost:8000")
DATA_GEN_VLLM_MODEL = _env_optional_str("DATA_GEN_VLLM_MODEL", "meta-llama/Llama-2-70b")

# ---- OpenAI Configuration (API-based) ----
OPENAI_API_KEY = _env_optional_str("OPENAI_API_KEY", "")
DATA_GEN_OPENAI_MODEL = _env_optional_str("DATA_GEN_OPENAI_MODEL", "gpt-4o")

# ---- Google Gemini Configuration (API-based) ----
GOOGLE_API_KEY = _env_optional_str("GOOGLE_API_KEY", "")
DATA_GEN_GEMINI_MODEL = _env_optional_str("DATA_GEN_GEMINI_MODEL", "gemini-1.5-pro")

# Temperature for data generation (higher = more diversity)
DATA_GEN_TEMPERATURE = float(_env_optional_str("DATA_GEN_TEMPERATURE", "0.7"))

# Max tokens for data generation
DATA_GEN_MAX_TOKENS = _env_optional_int("DATA_GEN_MAX_TOKENS", 2048)

# Timeout for LLM requests (seconds)
DATA_GEN_TIMEOUT = _env_optional_int("DATA_GEN_TIMEOUT", 300)

# ============================================================================
# Data Augmentation Configuration
# ============================================================================

# Number of paraphrases per question
PARAPHRASE_COUNT = _env_optional_int("PARAPHRASE_COUNT", 8)

# Number of KB-driven questions per KB chunk
KB_QUESTIONS_PER_CHUNK = _env_optional_int("KB_QUESTIONS_PER_CHUNK", 3)

# Include adversarial "I don't know" questions
INCLUDE_ADVERSARIAL = _env_optional_str("INCLUDE_ADVERSARIAL", "true").lower() == "true"

# ============================================================================
# Validation Configuration
# ============================================================================

# Minimum score (1-5) to accept a generated pair
VALIDATION_MIN_SCORE = _env_optional_int("VALIDATION_MIN_SCORE", 4)

# Enable semantic deduplication
ENABLE_DEDUPLICATION = _env_optional_str("ENABLE_DEDUPLICATION", "true").lower() == "true"

# Cosine similarity threshold for deduplication
DEDUP_SIMILARITY_THRESHOLD = float(_env_optional_str("DEDUP_SIMILARITY_THRESHOLD", "0.85"))

# ============================================================================
# Fine-tuning Configuration
# ============================================================================

# LoRA hyperparameters
LORA_RANK = _env_optional_int("LORA_RANK", 32)
LORA_ALPHA = _env_optional_int("LORA_ALPHA", 64)
LORA_DROPOUT = float(_env_optional_str("LORA_DROPOUT", "0.05"))

# Training hyperparameters
TRAINING_EPOCHS = _env_optional_int("TRAINING_EPOCHS", 2)
LEARNING_RATE = float(_env_optional_str("LEARNING_RATE", "2e-4"))
BATCH_SIZE = _env_optional_int("BATCH_SIZE", 4)
GRADIENT_ACCUMULATION_STEPS = _env_optional_int("GRADIENT_ACCUMULATION_STEPS", 4)
WARMUP_STEPS = _env_optional_int("WARMUP_STEPS", 100)
MAX_SEQ_LENGTH = _env_optional_int("MAX_SEQ_LENGTH", 2048)
SAVE_STEPS = _env_optional_int("SAVE_STEPS", 500)

# Model configuration
BASE_MODEL = _env_optional_str("BASE_MODEL", "Qwen/Qwen2.5-14B")
QUANTIZATION = _env_optional_str("QUANTIZATION", "4bit")  # "4bit" or "8bit"

# ============================================================================
# Input/Output Paths
# ============================================================================

HUMAN_QA_EXCEL = KB_DIR / "Human validated 50 questions.xlsx"
KB_VOCABULARY = KB_DIR / "kb_vocabulary.xlsx"
KB_DATA = OUTPUTS_DIR / "Normalized_kb.xlsx"

# Output files
AUGMENTED_QUESTIONS_JSONL = FINETUNING_OUTPUTS_DIR / "augmented_questions.jsonl"
VALIDATED_QUESTIONS_JSONL = FINETUNING_OUTPUTS_DIR / "validated_questions.jsonl"
TRAIN_DATASET_JSONL = FINETUNING_OUTPUTS_DIR / "train_dataset.jsonl"
VAL_DATASET_JSONL = FINETUNING_OUTPUTS_DIR / "val_dataset.jsonl"
CHECKPOINT_DIR = FINETUNING_OUTPUTS_DIR / "checkpoints"

# Create checkpoint directory
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
