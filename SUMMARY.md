# Advanced Programming for Data Science — Course Summary

Topic: **Hugging Face & Supervised Fine-Tuning of LLMs**

---

## 1. Introduction to Hugging Face

**Hugging Face** is an open-source AI platform and community. It provides:

- **Model Hub** – thousands of pretrained models (BERT, GPT-2, T5, LLaMA, Qwen, …) ready to download.
- **Datasets Hub** – public datasets loadable with `load_dataset()`.
- **Libraries** – `transformers` (models, tokenizers, trainers), `datasets`, `peft` (LoRA), `trl` (`SFTTrainer`), `accelerate`, `bitsandbytes` (quantization).
- **Spaces** – host and share ML demos.

**How it relates to fine-tuning:** Hugging Face gives us the full pipeline. We load a pretrained model from the Hub, prepare data with `datasets`, then train with `Trainer`/`SFTTrainer` plus PEFT/LoRA. The fine-tuned model can be pushed back to the Hub.

---

## 2. Fine-Tuning

**Fine-tuning** adapts a **pretrained model** to a new, specific task by continuing its training on a **smaller, task-specific dataset**. It reuses the knowledge learned from large-scale data and adjusts the model's parameters, so the new task performs better without training from scratch.

---

## 3. Model & Preprocessor Setup

### Model Loader
Loads a pretrained model's architecture and weights into memory for inference or fine-tuning.

| Type | Description | Example |
|------|-------------|---------|
| **Model-specific** | Tied to a known architecture | `BertForSequenceClassification`, `GPT2LMHeadModel`, `T5ForConditionalGeneration` |
| **AutoModel** | Picks the right architecture automatically from the model ID | `AutoModelForCausalLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification` |

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### Processor Loader
`AutoProcessor` / `AutoTokenizer` automatically selects the correct preprocessor for the model.
Processor types: **Tokenizer, Image Processor, Audio Processor, Multimodal Processor, Audio-Text Processor**.

### Quantization
Reduces the precision of weights (e.g., FP32 → FP16 / INT8 / 4-bit) to make models **smaller, faster and less memory-hungry**. Configured with `BitsAndBytesConfig`:

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # or load_in_8bit=True
    bnb_4bit_use_double_quant=True,        # extra quantization for efficiency
    bnb_4bit_quant_type="nf4",             # quantization type
    bnb_4bit_compute_dtype=torch.float16,  # dtype used in forward pass
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
```

---

## 4. Data Preparation

Data must be structured to match the tokenizer, model architecture and training pipeline.

1. **Raw format** – JSON / CSV / text. Key-value JSON works best (easy to load with `load_dataset`).
2. **Conversation format** – organize data as multi-turn messages:
   ```python
   [{"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}]
   ```
   *To identify the expected format: check the model card on Hugging Face, print the chat template, or check the model's training dataset.*
3. **Chat template** – `tokenizer.apply_chat_template()` converts the conversation into the single text string the chat model expects.
4. **Data collator** – builds batches: tokenization, **padding**, masks/labels, and sometimes applying chat templates. It comes from `transformers`/`unsloth` or a custom function and is **passed to the trainer**, not called manually.

---

## 5. Supervised Trainer Setup

### PEFT (Parameter-Efficient Fine-Tuning)
Fine-tunes only a **small subset of parameters** instead of the whole model, so training is faster, uses less memory and is cheaper.

### LoRA (Low-Rank Adaptation)
Freezes the original weights `W` and adds two small trainable matrices `A` and `B` (rank `r` ≪ model size):

$$W' = W + AB$$

Example: `W` is 512×512 (262,144 params). With `r = 8`: `A` (512×8) + `B` (8×512) = **8,192 params**, about **97% fewer** trainable parameters.

```python
from peft import LoraConfig
lora_config = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],   # find names with print(model)
    task_type="CAUSAL_LM",
)
```

### Trainer
Pass the model, processor, data collator, configs (LoRA + training arguments) and train/eval datasets to the trainer, then start training:

```python
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model=model, args=SFTConfig(output_dir="outputs", report_to="wandb"),
    train_dataset=train_ds, eval_dataset=eval_ds,
    peft_config=lora_config, data_collator=collator,
)
trainer.train()
```

---

## 6. Training Diagnosis

| Observation | Meaning |
|-------------|---------|
| Training loss decreasing | Model is learning |
| Very low train loss, val loss much higher | **Overfitting** (memorizing training data) |
| Both train & val loss high | **Underfitting** (not learned enough) |
| Both decrease and stay close | Good generalization |

- **Training loss**: how well the model fits the training data.
- **Validation loss**: how well the model generalizes to unseen data.
- **Weights & Biases (WandB)**: set `report_to="wandb"` and call `wandb.init()` before `trainer.train()`. Losses and metrics are then logged to an online dashboard automatically.

---

## 7. Model Checkpoints

A **checkpoint** is a saved state of the model during training: weights, optimizer state, step and loss. Checkpoints let you:

- Resume interrupted training
- Save progress and avoid retraining from scratch
- Compare stages and pick the best model
- Deploy for inference

Load a checkpoint directly, or **apply the LoRA adapter on top of its base model**:

```python
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained(base_model_id)
model = PeftModel.from_pretrained(base, "outputs/checkpoint-500")
```

---

## Key Takeaways

- Hugging Face provides the whole ecosystem: models, datasets, and training tools.
- Fine-tuning = pretrained model + task-specific data → specialized model.
- **Quantization + LoRA (QLoRA)** make LLM fine-tuning possible on limited GPUs.
- Correct **data formatting (conversation + chat template)** is essential for chat models.
- Watch **train vs. validation loss** to catch over/underfitting, and save **checkpoints**.
