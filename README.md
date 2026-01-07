
# Igbo Language Model Training with AWS SageMaker

> **📖 New to this project?** Start with [INDEX.md](INDEX.md) for complete navigation and documentation structure.

---

Train a world-class Igbo-English translation model using Meta's NLLB dataset and AWS SageMaker.

## 📋 Overview

This project fine-tunes Mistral 7B on 6.1 million Igbo-English sentence pairs using LoRA (Low-Rank Adaptation) for efficient training. The resulting model can:
- Translate between Igbo and English
- Understand and generate natural Igbo text
- Run locally on consumer hardware (16GB RAM)
- Be integrated into mobile applications

**Training Stats:**
- Dataset: NLLB (No Language Left Behind) - 6.1M sentence pairs
- Training Examples: 19.5M (4 formats per pair)
- Model: Mistral-7B-v0.1 with LoRA adapters
- Training Time: ~150-180 hours (6-7 days)
- Cost: ~$70-80 (with spot instances)
- Hardware: AWS ml.g5.xlarge (NVIDIA A10G GPU)

## 🎯 Project Goals

- Create a high-quality Igbo language model for educational purposes
- Enable Igbo language learning for children
- Provide offline translation capabilities
- Support iOS app integration for OCR and translation

## 📁 Repository Structure

```
igbo-llm-training/
├── README.md                           # This file
├── docs/
│   ├── 01-setup.md                     # AWS account and quota setup
│   ├── 02-data-preparation.md          # Dataset download and preparation
│   ├── 03-sagemaker-training.md        # SageMaker training setup
│   ├── 04-monitoring.md                # Monitoring and troubleshooting
│   └── 05-deployment.md                # Model deployment options
├── scripts/
│   ├── train_igbo_model.py             # SageMaker training script
│   ├── igbo-train.ipynb                # Jupyter notebook for launching training
│   ├── prepare_nllb_training.py        # Prepare training data (reference)
│   └── (other utility scripts)
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore file
```

## 🚀 Quick Start

### Prerequisites

- AWS account with SageMaker access
- Python 3.10+
- AWS CLI configured
- 16GB+ RAM for local inference (optional)

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/igbo-llm-training.git
cd igbo-llm-training
```

### 2. Set Up AWS

```bash
# Configure AWS credentials
aws configure

# Request SageMaker quotas (if needed)
# See docs/01-setup.md for details
```

### 3. Download and Prepare Data

```bash
# Download NLLB dataset
chmod +x scripts/download_nllb.sh
./scripts/download_nllb.sh

# Prepare training data
python3 scripts/prepare_nllb_training.py
```

### 4. Upload to S3

```bash
# Create S3 bucket in us-east-1
aws s3 mb s3://your-bucket-name --region us-east-1

# Upload training data
aws s3 sync ./datasets/processed-nllb s3://your-bucket-name/datasets/nllb/
```

### 5. Launch Training

```bash
# Option 1: Using Python script
python3 scripts/launch_sagemaker_training.py

# Option 2: Using SageMaker Studio Notebook
# Upload notebooks/sagemaker_training.ipynb to SageMaker Studio
# Run cells in order
```

### 6. Monitor Training

```bash
# Check job status
aws sagemaker describe-training-job \
  --training-job-name your-job-name \
  --region us-east-1

# View logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### 7. Download Trained Model

```bash
# After training completes
aws s3 cp s3://your-bucket-name/models/igbo-llm/[job-name]/output/model.tar.gz ./

# Extract
tar -xzf model.tar.gz
```

## 💰 Cost Breakdown

| Item | Cost |
|------|------|
| SageMaker Training (spot) | $63-76 |
| S3 Storage (5GB) | $0.50/month |
| Data Transfer | $0.10 (one-time) |
| **Total First Run** | **~$70-80** |

## 📊 Training Configuration

```python
# Model
model_name = 'mistralai/Mistral-7B-v0.1'
base_parameters = 7 billion

# LoRA Configuration
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
trainable_parameters = ~8.4 million (0.12% of base)

# Training Hyperparameters
epochs = 3
learning_rate = 2e-4
batch_size = 2 (per device)
gradient_accumulation_steps = 8
effective_batch_size = 16
max_length = 512 tokens

# Hardware
instance_type = ml.g5.xlarge
gpu = NVIDIA A10G (24GB VRAM)
spot_instances = True (70% discount)
```

## 🎓 Dataset Information

**NLLB (No Language Left Behind)**
- Source: Meta AI / OPUS
- Sentence Pairs: 6,110,033
- Languages: Igbo (ig) ↔ English (en)
- Quality: Professional translations
- Domain: Diverse (news, literature, technical, conversational)

**Training Format (4 examples per pair):**
1. Direct translation: Igbo → English
2. Reverse translation: English → Igbo
3. Question format: "What does X mean in English?"
4. Question format: "How do you say Y in Igbo?"

## 🖥️ Local Deployment

After training, run the model locally with Ollama:

```bash
# Convert to GGUF format (quantized for efficiency)
python3 convert_to_gguf.py --model ./model --outfile igbo-model-q4.gguf --outtype q4_k_m

# Create Ollama Modelfile
cat > Modelfile << EOF
FROM igbo-model-q4.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are an expert Igbo-English translator."
EOF

# Import to Ollama
ollama create igbo-teacher -f Modelfile

# Test translation
ollama run igbo-teacher "Translate to Igbo: Hello, how are you?"
```

**System Requirements for Local Inference:**
- RAM: 16GB minimum (8GB for Q4 model + 8GB for OS/apps)
- Storage: 5GB for Q4 quantized model
- CPU: Any modern CPU (M1/M2 Mac, Intel i5+, AMD Ryzen 5+)
- GPU: Optional (speeds up inference 5-10x)

## 🆘 Troubleshooting

### Issue: ResourceLimitExceeded

**Problem:** No quota for GPU instances

**Solution:**
```bash
# Request quota increase
# Go to: https://console.aws.amazon.com/servicequotas/
# Search: "ml.g5.xlarge for spot training job usage"
# Request: 1 instance
# Wait: 24-48 hours for approval
```

### Issue: AccessDenied to S3

**Problem:** SageMaker role lacks S3 permissions

**Solution:**
1. Go to IAM Console
2. Find role: `AmazonSageMaker-ExecutionRole-*`
3. Attach policy: `AmazonS3FullAccess`

### Issue: CUDA Out of Memory

**Problem:** Batch size too large for GPU

**Solution:**
```python
# Reduce batch size
per_device_train_batch_size = 1  # Instead of 2
gradient_accumulation_steps = 16  # Instead of 8
```

### Issue: Spot Instance Interrupted

**Problem:** AWS reclaimed spot instance

**Solution:** Training automatically resumes from last checkpoint (no action needed)

## 📚 Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [NLLB Dataset](https://opus.nlpl.eu/NLLB.php)
- [Ollama Documentation](https://github.com/ollama/ollama)

## 📄 License

MIT License - See LICENSE file for details

## 📚 Complete Documentation

This README provides a quick overview. For comprehensive step-by-step guides:

- **[INDEX.md](INDEX.md)** - Complete documentation index with navigation
- **[MEMORY_OPTIMIZATION_CHANGES.md](MEMORY_OPTIMIZATION_CHANGES.md)** - Critical OOM fixes
- **[docs/](docs/)** - Detailed guides for each phase:
  - [01-setup.md](docs/01-setup.md) - AWS setup (30-60 min)
  - [02-data-preparation.md](docs/02-data-preparation.md) - Data prep (2-3 hours)
  - [03-sagemaker-training.md](docs/03-sagemaker-training.md) - Training (7-8 days)
  - [04-monitoring.md](docs/04-monitoring.md) - Monitoring tools
  - [05-deployment.md](docs/05-deployment.md) - Deployment options

## 🙏 Acknowledgments

- Meta AI for the NLLB dataset
- Mistral AI for the base model
- AWS for SageMaker infrastructure
- OPUS for dataset hosting
- Igbo language community

---

**Built with ❤️ for Igbo language preservation and education** 🇳🇬