# Deployment Guide

This guide covers deploying your trained Igbo language model for local inference on a Mac Mini M2 with 16GB RAM.

## Overview

After training completes, you'll have a LoRA adapter (~80MB) that works with the base Mistral-7B model. This guide walks you through:

- **Downloading** your trained model from S3
- **Converting** the model to GGUF format for efficient inference
- **Setting up** Ollama on your Mac Mini
- **Testing** the model for translations

**Requirements:**
- Mac Mini M2 with 16GB RAM
- 8GB free disk space
- macOS 12+ (Sonoma or later recommended)
- Python 3.9+ and pip installed

---

## Step 1: Download Trained Model

### From S3

Download your trained model artifacts from S3 using the AWS CLI. You'll need your bucket name and training job name.

The downloaded archive contains:
- `adapter_config.json` - LoRA configuration
- `adapter_model.bin` or `adapter_model.safetensors` - LoRA weights (~80MB)
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, etc.)
- `training_args.bin` - Training configuration

Extract the archive to access the model files.

---

## Step 2: Install Ollama

Ollama makes it easy to run LLMs locally on your Mac. Install it by downloading from [ollama.com](https://ollama.com/download) or using the installation script.

After installation, verify it works by checking the version in Terminal.

---

## Step 3: Convert Model to GGUF Format

To use the model with Ollama, you need to convert it from PyTorch format to GGUF format. This process involves several steps:

### Prerequisites

You'll need to install:
- `llama.cpp` (for conversion tools)
- Python packages: `torch`, `transformers`, `peft`, `huggingface-hub`

### Process Overview

1. **Download the base Mistral model** - The base model (~14GB) needs to be downloaded from Hugging Face
2. **Merge LoRA adapter** - Combine your LoRA adapter with the base model into a single merged model
3. **Convert to GGUF** - Convert the merged PyTorch model to GGUF format
4. **Quantize** - Reduce model size for efficient inference on 16GB RAM

### Quantization Options

For a Mac Mini with 16GB RAM, you have these options:

| Format | Size | Quality | RAM Usage | Recommendation |
|--------|------|---------|-----------|----------------|
| Q5_K_M | 5GB | 97% | ~7GB | ✅ Best balance |
| Q4_K_M | 4GB | 95% | ~6GB | ✅ Fastest |
| Q8_0 | 7.5GB | 99% | ~10GB | Higher quality |
| Q3_K_M | 3GB | 90% | ~5GB | If space is tight |

**Recommended:** Q4_K_M or Q5_K_M for optimal performance on 16GB RAM.

The conversion process requires:
- Running a merge script to combine LoRA with the base model
- Using `llama.cpp` conversion tools to create GGUF files
- Running quantization to reduce file size

Detailed conversion scripts and commands are available in the project's conversion utilities.

---

## Step 4: Create Ollama Model

Once you have your quantized GGUF file, create an Ollama model using a Modelfile. The Modelfile configures:

- Model path (your GGUF file)
- Prompt template for translation tasks
- Generation parameters (temperature, top_p, top_k, repeat_penalty)
- System prompt describing the model's role as an Igbo-English translator

After creating the Modelfile, use `ollama create` to register your model with Ollama.

---

## Step 5: Test the Model

Test your model with simple translation prompts:

- **English to Igbo:** "Translate to Igbo: Good morning, how are you?"
- **Igbo to English:** "Translate to English: Ọ bụ nnọọ mma ịhụ gị"

The model should respond with accurate translations. Adjust the Modelfile parameters if you need different behavior.

---

## Expected Performance on Mac Mini M2

With Q4_K_M quantization (4GB model):

- **RAM usage:** 6-8GB
- **Speed:** 20-30 tokens/second
- **First token latency:** 500-800ms
- **Subsequent tokens:** 33-50ms each

These numbers may vary based on system load and model quantization level.

---

## Troubleshooting

### Model Too Large

If you're running out of memory, use more aggressive quantization (Q3_K_M or Q2_K). This reduces quality slightly but allows the model to run on less RAM.

### Slow Inference

To improve speed:
- Use Q4_K_M quantization (faster than Q5)
- Reduce max_tokens in your requests
- Close other applications to free up RAM
- Ensure you're using the M2's Neural Engine (Ollama does this automatically)

### Poor Translation Quality

If translations aren't accurate:
- Try Q5_K_M or Q8_0 quantization (less compression = better quality)
- Check that training completed successfully
- Verify your test data matches the training data distribution
- Adjust the prompt template in your Modelfile

### Installation Issues

If you encounter problems:
- Ensure Python 3.9+ is installed: `python3 --version`
- Update pip: `pip3 install --upgrade pip`
- Install required packages: `pip3 install torch transformers peft huggingface-hub`
- Check Ollama is running: `ollama list`

---

## Cost Comparison

### Local Deployment (Mac Mini M2)

| Item | Cost |
|------|------|
| Hardware | $599 (one-time) |
| Electricity | ~$5/month |
| Maintenance | $0 |
| **Total first year** | **~$660** |

### Cloud Alternatives

For comparison, cloud deployment options:
- SageMaker ml.g5.xlarge: $1,016/month
- SageMaker ml.inf2.xlarge: $547/month
- HuggingFace Inference: $432/month

**Recommendation:** For personal use and development, local deployment on Mac Mini is the most cost-effective option.

---

## Next Steps

1. ✅ Convert model to GGUF format
2. ✅ Set up Ollama on Mac Mini
3. ✅ Test translations
4. ✅ Integrate into iOS app
5. ✅ Start teaching Igbo! 🇳🇬

---

## Resources

- **Ollama Documentation:** https://ollama.com/docs
- **llama.cpp Repository:** https://github.com/ggerganov/llama.cpp
- **GGUF Format Specs:** https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Hugging Face Quantization Guide:** https://huggingface.co/docs/transformers/main/en/quantization

---

**Your Igbo language model is ready to use!** 🎉🇳🇬

Share it with your kids and help preserve the Igbo language! 🌍
