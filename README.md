Open the notebooks from google colab if you have rendering issues by following links:
'finetune_llama2.ipynb' : https://colab.research.google.com/drive/1pQDDgOoBQFpIdwfCA6Tna-LdFvQ2XVBb?usp=sharing
'merge_llama.ipynb' : https://colab.research.google.com/drive/1Lo_5z5yXsmTh_ymgPcxSZ8qRqC28sSuo?usp=sharing 

# LLaMA-2 Fine-tuning and Deployment

Code for fine-tuning LLaMA-2 model on custom datasets using QLoRA and deploying the resulting model.

## Notebooks

### 1. `finetune_llama.ipynb`
- Model configuration and initialization
- Dataset preparation and splitting
- QLoRA and training setup
- Fine-tuning execution

**Key Components:**
- QLoRA parameters configuration
- Dataset splitting by paper DOIs
- 4-bit quantization setup
- Training configuration with LoRA

### 2. `merge_llama.ipynb`
- Model merging and deployment utilities
- Text generation interface

**Key Features:**
- LoRA weight merging with base model
- Model verification functionality
- Text generation class with customizable parameters
- Efficient inference setup

## Setup and Usage

1. **Fine-tuning:**
```python
# Configure model parameters
model_name = "NousResearch/Llama-2-7b-hf"
dataset_name = "jamescalam/ai-arxiv2-chunks"

# Run training
trainer.train()
```

2. **Merging Models:**
```python
merged_model, tokenizer = merge_lora_with_base(
    base_model_name=BASE_MODEL,
    lora_path=LORA_PATH,
    save_path=SAVE_PATH
)
```

3. **Text Generation:**
```python
generator = TextGenerator(model_path="path/to/model")
output = generator.generate(
    prompt="Your prompt",
    temperature=0.7
)
```

## Requirements
- PyTorch
- Transformers
- PEFT
- bitsandbytes
- pandas
- scikit-learn

## GPU Requirements
- CUDA-compatible GPU recommended
- Supports bfloat16 training on Ampere+ GPUs
