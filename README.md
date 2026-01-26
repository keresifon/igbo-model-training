
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

**Production Training Results:**
- Dataset: NLLB (No Language Left Behind) - 6.1M sentence pairs
- Training Examples: 19.5M (bidirectional augmentation)
- Model: Mistral-7B-v0.1 with LoRA adapters
  - Trainable Parameters: 6.8M (0.094% of full model)
- Training Time: 168 hours (7 days continuous)
- Cost: $237 (on-demand instances for guaranteed completion)
- Hardware: AWS ml.g5.xlarge (NVIDIA A10G GPU, 24GB VRAM)
- GPU Utilization: 90-95% maintained throughout
- Throughput: 11.56 iterations/second

## 🎯 Project Goals

- Create a high-quality Igbo language model for educational purposes
- Enable Igbo language learning for children
- Provide offline translation capabilities
- Support mobile app integration for OCR and translation

## 📊 Project Status

**Current Status:** ✅ Production training completed (January 2026)

### Latest Training Run
- **Job Name:** igbo-nllb-fixed-2026-01-24
- **Duration:** 168 hours (completed January 31, 2026)
- **Status:** Successfully executed with lessons learned
- **Checkpoints:** ~30 checkpoints saved (every 5000 steps)
- **Cost:** $237

### What This Repository Demonstrates
This project represents complete production-scale ML engineering experience:
- ✅ End-to-end AWS SageMaker pipeline (5 AWS services)
- ✅ 168-hour continuous training execution
- ✅ Real-world cost-reliability tradeoffs
- ✅ Production troubleshooting and recovery
- ✅ Infrastructure capacity planning (Service Quotas management)
- ✅ Defensive checkpoint strategies
- ✅ Complete documentation and knowledge transfer

### Key Production Learnings

**1. Cost vs Reliability Decision**
- Initial attempt: Spot instances ($0.42/hr, 70% savings)
- Challenge: Capacity interruptions after 41 hours
- Decision: Switched to on-demand ($1.41/hr)
- **Learning:** For 7-day jobs, reliability > cost optimization

**2. Checkpoint Configuration (Critical)**
- First run: Completed 168 hours but only preserved initial checkpoint
- Root cause: Checkpoint save location vs S3 sync mismatch
- Impact: Lost 168 hours of training progress
- Solution: Implemented validation callbacks + proper sync configuration
- **Learning:** Checkpoint strategies must be validated EARLY
- Full analysis: See `docs/LESSONS_LEARNED.md`

**3. Infrastructure Planning**
- Proactively requested AWS Service Quotas extension (5→7 days)
- Approved within 24 hours
- **Learning:** Advance capacity planning prevents delays


## ⭐ Featured Files

Key files to get started:

- **[scripts/train_igbo_model.py](scripts/train_igbo_model.py)** - Main SageMaker training script with LoRA fine-tuning
- **[scripts/igbo-train.ipynb](scripts/igbo-train.ipynb)** - Jupyter notebook for launching training jobs
- **[scripts/prepare_nllb_training.py](scripts/prepare_nllb_training.py)** - Data preparation script for NLLB dataset
- **[docs/03-sagemaker-training.md](docs/03-sagemaker-training.md)** - Complete training guide with hyperparameters
- **[docs/05-deployment.md](docs/05-deployment.md)** - Deployment guide for Mac Mini M2 with Ollama
- **[MEMORY_OPTIMIZATION_CHANGES.md](MEMORY_OPTIMIZATION_CHANGES.md)** - Critical fixes for OOM errors

## 📁 Repository Structure

```
igbo-llm-training/
├── README.md                           # This file
├── LICENSE                             # MIT License
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
├── datasets/                           # Processed datasets
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

# CRITICAL: Checkpoint Configuration
save_strategy = "steps"
save_steps = 5000  # Save every 5000 steps
save_total_limit = None  # IMPORTANT: Keep all checkpoints!

# Output Configuration
output_dir = "/opt/ml/output/data"  # Auto-uploads to S3 at completion
# Note: checkpoint_s3_uri may not work reliably with HuggingFace estimator
# Checkpoints in output_dir will upload when training completes

```

## ☁️ AWS Services

This project uses the following AWS services:

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **SageMaker Training** | Model training on GPU instances | ml.g5.xlarge (NVIDIA A10G, 24GB VRAM) |
| **S3** | Data storage and model artifacts | ~5GB training data, ~80MB model output |
| **CloudWatch** | Training logs and metrics | Automatic logging, custom metrics |
| **IAM** | Access control and permissions | SageMaker execution role with S3 access |
| **Service Quotas** | Resource limits management | ml.g5.xlarge quota request required |

## 💰 Cost Breakdown

### Actual Production Costs
| Item | Cost | Notes |
|------|------|-------|
| SageMaker Training (on-demand) | $237 | 168 hours × $1.41/hour |
| S3 Storage (5GB) | $0.50/month | Data + checkpoints |
| Data Transfer | $0.10 | One-time upload |
| Service Quotas Extension | $0 | Free, 24hr approval |
| **Total First Run** | **~$238** | Guaranteed completion |

### Cost Considerations
- **Spot Instances:** 70% cheaper ($63-76 for 7 days) BUT risk interruptions
- **On-Demand:** Higher cost BUT guaranteed 7-day completion
- **Recommendation:** Use on-demand for jobs >3 days to avoid restarts

For detailed AWS setup instructions, see [docs/01-setup.md](docs/01-setup.md).

## 🎓 Dataset Information

**NLLB (No Language Left Behind)**
- Source: Meta AI / OPUS
- Sentence Pairs: 6,110,033
- Languages: Igbo (ig) ↔ English (en)
- Quality: Professional translations
- Domain: Diverse (news, literature, technical, conversational)
- License: ODC Attribution License (ODC-By)

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
### Issue: Checkpoints Not Syncing During Training

**Problem:** Training completes but only first checkpoint preserved

**Root Cause:** Mismatch between checkpoint save location and S3 sync configuration

**Symptoms:**
- Training runs successfully for hours/days
- CloudWatch shows "CHECKPOINT SAVED" messages
- Only checkpoint-1000 or checkpoint-5000 appears in S3
- Later checkpoints missing

**Solution:**
```python
# In training script (train_igbo_model_FIXED.py):
training_args = TrainingArguments(
    output_dir="/opt/ml/output/data",  # Use this, not /opt/ml/checkpoints
    save_strategy="steps",
    save_steps=5000,
    save_total_limit=None,  # CRITICAL: Don't delete old checkpoints!
    # Checkpoints will upload to S3 when training completes
)
Prevention:

Use save_total_limit=None to preserve all checkpoints
Add checkpoint validation callback (see train_igbo_model_FIXED.py)
Test checkpoint saving in first 30 minutes of training
Verify checkpoint upload to S3 after job completion
Reference: See docs/LESSONS_LEARNED.md for complete analysis


## 🤝 Ethics & Responsible AI

### Purpose & Use Cases
This model is designed for **educational and language preservation purposes**:
- Teaching Igbo language to children and learners
- Preserving Igbo cultural heritage through technology
- Enabling offline translation for communities with limited internet access
- Supporting academic research in African languages

### Limitations & Considerations

**Data Quality:**
- Model quality depends on the NLLB dataset, which may contain biases or errors
- Translations may not capture all cultural nuances or regional variations
- Professional review recommended for critical translations

**Bias & Fairness:**
- The model reflects patterns in the training data
- May perpetuate existing biases in the source dataset
- Should not be used for automated decision-making without human oversight

**Privacy:**
- Model does not store or transmit user input data
- Local deployment option ensures data privacy
- No user data is collected or shared

**Responsible Deployment:**
- ✅ Use for educational and language learning purposes
- ✅ Use for personal translation assistance
- ✅ Use for cultural preservation initiatives
- ❌ Do not use for automated content moderation
- ❌ Do not use for critical medical or legal translations without human review
- ❌ Do not use to replace human translators in sensitive contexts

### Attribution
- **Dataset:** NLLB (Meta AI) - ODC Attribution License
- **Base Model:** Mistral-7B-v0.1 (Mistral AI) - Apache 2.0
- **Training Infrastructure:** AWS SageMaker

We acknowledge and respect the contributions of the Igbo language community and the organizations that made this project possible.

## 📚 Additional Resources

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [NLLB Dataset](https://opus.nlpl.eu/NLLB.php)
- [Ollama Documentation](https://github.com/ollama/ollama)

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

**Note:** The NLLB dataset is licensed under ODC Attribution License (ODC-By). The base Mistral-7B model is licensed under Apache 2.0.

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