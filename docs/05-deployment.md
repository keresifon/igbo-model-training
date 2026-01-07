# Deployment Guide

This guide covers deploying your trained Igbo language model for inference, both locally and in production.

## Overview

After training completes, you'll have a LoRA adapter (~80MB) that works with the base Mistral-7B model. This guide covers:

- **Local deployment** with Ollama (Mac Mini, 16GB RAM)
- **Model conversion** (PyTorch → GGUF)

---

## Step 1: Download Trained Model

### From S3

```bash
# List available models
aws s3 ls s3://your-bucket/models/igbo-llm/

# Download model artifacts
JOB_NAME="your-training-job-name"
aws s3 cp s3://your-bucket/models/igbo-llm/${JOB_NAME}/output/model.tar.gz .

# Extract
tar -xzf model.tar.gz
```

### Model Contents

```
model/
├── adapter_config.json      # LoRA configuration
├── adapter_model.bin         # LoRA weights (~80MB)
├── adapter_model.safetensors # Alternative format
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── training_args.bin
```

---

## Step 2: Local Deployment with Ollama

### Overview

Ollama makes it easy to run LLMs locally. We'll convert the model to GGUF format and quantize it for efficient inference.

**Requirements:**
- Mac Mini M1/M2 with 16GB RAM ✅
- 8GB free disk space
- macOS 12+ or Linux

### Install Ollama

```bash
# macOS
curl -fsSL https://ollama.com/install.sh | sh

# Or download from: https://ollama.com/download

# Verify installation
ollama --version
```

### Convert Model to GGUF

#### Step 2.1: Install Dependencies

```bash
# Clone llama.cpp (for conversion)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install Python dependencies
pip3 install -r requirements.txt

# Build
make
```

#### Step 2.2: Merge LoRA with Base Model

```bash
# Download base Mistral model
mkdir -p models/mistral-7b-base
huggingface-cli download mistralai/Mistral-7B-v0.1 \
    --local-dir models/mistral-7b-base \
    --local-dir-use-symlinks False

# Merge LoRA adapter with base model
python scripts/merge_lora.py \
    --base-model models/mistral-7b-base \
    --lora-adapter path/to/your/adapter \
    --output models/igbo-mistral-merged
```

**Merge script** (`scripts/merge_lora.py`):

```python
#!/usr/bin/env python3
"""Merge LoRA adapter with base model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def merge_lora(base_model_path, lora_path, output_path):
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("Merging...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("✓ Merge complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--lora-adapter', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    merge_lora(args.base_model, args.lora_adapter, args.output)
```

#### Step 2.3: Convert to GGUF

```bash
# Convert merged model to GGUF FP16
python llama.cpp/convert.py \
    models/igbo-mistral-merged \
    --outfile models/igbo-mistral.gguf \
    --outtype f16

# File size: ~14GB
```

#### Step 2.4: Quantize for Efficiency

```bash
# Quantize to Q4_K_M (4-bit, ~4GB)
./llama.cpp/quantize \
    models/igbo-mistral.gguf \
    models/igbo-mistral-q4.gguf \
    Q4_K_M

# Or Q5_K_M for better quality (~5GB)
./llama.cpp/quantize \
    models/igbo-mistral.gguf \
    models/igbo-mistral-q5.gguf \
    Q5_K_M
```

**Quantization Options:**

| Format | Size | Quality | Speed | RAM |
|--------|------|---------|-------|-----|
| F16 | 14GB | 100% | Slow | 16GB+ |
| Q8_0 | 7.5GB | 99% | Medium | 10GB |
| Q5_K_M | 5GB | 97% | Fast | 7GB ✅ |
| Q4_K_M | 4GB | 95% | Fastest | 6GB ✅ |
| Q3_K_M | 3GB | 90% | Fastest | 5GB |

**Recommended for Mac Mini 16GB:** Q4_K_M or Q5_K_M

### Create Ollama Modelfile

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./models/igbo-mistral-q4.gguf

TEMPLATE """### Instruction:
{{.Prompt}}

### Response:
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """You are an expert Igbo-English translator. Translate accurately while preserving cultural context and idiomatic expressions."""
EOF

# Create Ollama model
ollama create igbo-translator -f Modelfile
```

### Test the Model

```bash
# Test translation EN → IG
ollama run igbo-translator "Translate to Igbo: Good morning, how are you?"

# Expected output:
# Ụtụtụ ọma, kedu ka ị mere?

# Test translation IG → EN
ollama run igbo-translator "Translate to English: Ọ bụ nnọọ mma ịhụ gị"

# Expected output:
# It's very nice to see you.
```

### Performance on Mac Mini

**Expected performance:**
- Quantization: Q4_K_M (4GB)
- RAM usage: 6-8GB
- Speed: 20-30 tokens/second
- First token: 500-800ms
- Subsequent tokens: 33-50ms each

---




---

## Model Evaluation

### Test Translation Quality

```python
#!/usr/bin/env python3
"""Evaluate translation quality"""

import ollama
from datasets import load_dataset

# Load test set
test_data = load_dataset('json', data_files='test_set.jsonl')

def evaluate_model(model_name, test_samples=100):
    correct = 0
    total = 0
    
    for example in test_data['train'].select(range(test_samples)):
        source = example['input']
        expected = example['output']
        instruction = example['instruction']
        
        # Get model translation
        response = ollama.generate(
            model=model_name,
            prompt=f"{instruction}\n{source}"
        )
        predicted = response['response'].strip()
        
        # Simple evaluation (you can use BLEU, METEOR, etc.)
        if predicted.lower() == expected.lower():
            correct += 1
        total += 1
        
        print(f"Source: {source}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        print(f"Match: {predicted.lower() == expected.lower()}\n")
    
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")
    
evaluate_model('igbo-translator', test_samples=50)
```

### Common Metrics

- **BLEU Score:** Measures n-gram overlap
- **METEOR:** Accounts for synonyms
- **chrF:** Character-level F-score
- **Human evaluation:** Best but slowest

---

## Troubleshooting

### Issue: Model too large for device

**Solution:** Use more aggressive quantization
```bash
# Q3_K_M (3GB) or Q2_K (2GB)
./quantize model.gguf model-q3.gguf Q3_K_M
```

### Issue: Slow inference

**Solutions:**
1. Use GPU acceleration (if available)
2. Reduce max_tokens
3. Use smaller quantization
4. Batch requests

### Issue: Poor translation quality

**Possible causes:**
1. Training didn't converge fully
2. Test data differs from training data
3. Quantization loss (try Q5 or Q8)
4. Prompt engineering needed

**Solutions:**
```python
# Better prompts
prompt = """### Instruction:
Translate the following English text to Igbo. Preserve the meaning and cultural context.

### English:
{text}

### Igbo:
"""
```

---

## Performance Optimization

### 1. Batch Processing

```python
# Process multiple translations together
texts = ["Hello", "Goodbye", "Thank you"]
prompts = [f"Translate to Igbo: {t}" for t in texts]

# Batch inference (if supported)
results = model.generate(prompts, batch_size=8)
```

### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def translate_cached(text, target_lang):
    return translate(text, target_lang)
```

### 3. Async Requests

```python
import asyncio

async def translate_many(texts):
    tasks = [translate_async(text) for text in texts]
    return await asyncio.gather(*tasks)
```

---

## Cost Comparison

### Local Deployment (Mac Mini)

| Item | Cost |
|------|------|
| Hardware | $599 (one-time) |
| Electricity | ~$5/month |
| Maintenance | $0 |
| **Total first year** | **~$660** |

### Cloud Deployment (24/7)

| Service | Monthly Cost |
|---------|--------------|
| SageMaker ml.g5.xlarge | $1,016 |
| SageMaker ml.inf2.xlarge | $547 |
| HuggingFace Inference | $432 |
| Lambda + EFS (low volume) | $10-50 |

**Recommendation:**
- **Development:** Local with Ollama
- **Production (high volume):** SageMaker
- **Production (low volume):** Lambda
- **Personal use:** Local Mac Mini

---

## Next Steps

1. ✅ Convert model to GGUF
2. ✅ Set up Ollama on Mac Mini
3. ✅ Test translations
4. ✅ Integrate into iOS app
5. ✅ Teach your kids Igbo! 🇳🇬

---

## Resources

- **Ollama Docs:** https://ollama.com/docs
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **GGUF Specs:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Model quantization guide:** https://huggingface.co/docs/transformers/main/en/quantization

---

**Your Igbo language model is ready to use!** 🎉🇳🇬

Share it with your kids and help preserve the Igbo language! 🌍
