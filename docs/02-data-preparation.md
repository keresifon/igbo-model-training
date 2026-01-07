# Data Preparation Guide

This guide covers preparing the NLLB dataset for training the Igbo language model.

## Overview

We use the NLLB (No Language Left Behind) dataset from OPUS, containing **6.1 million** English-Igbo sentence pairs.

**Dataset Statistics:**
- Source: OPUS NLLB corpus
- Sentence pairs: 6,145,395
- Training examples: 19,471,872 (4 formats per pair)
- Compressed size: ~500MB
- Uncompressed size: ~2.5GB
- Training file size: ~5GB (JSONL format)

---

## Step 1: Download NLLB Dataset

### Download from OPUS

```bash
# Create directory structure
mkdir -p ~/igbo/en-ig
cd ~/igbo/en-ig

# Download English-Igbo corpus
wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-ig.txt.zip

# Extract files
unzip en-ig.txt.zip
```

**Files extracted:**
- `NLLB.en-ig.en` - English sentences (6,145,395 lines)
- `NLLB.en-ig.ig` - Igbo sentences (6,145,395 lines)
- `README` - Dataset information

### Verify Data Integrity

```bash
# Count lines in each file
wc -l NLLB.en-ig.en
wc -l NLLB.en-ig.ig

# Should both show: 6145395

# Check alignment (line counts must match)
[ $(wc -l < NLLB.en-ig.en) -eq $(wc -l < NLLB.en-ig.ig) ] && echo "✅ Files aligned" || echo "❌ Alignment error"

# Sample first 5 pairs
paste NLLB.en-ig.en NLLB.en-ig.ig | head -5
```

**Expected output:**
```
6145395 NLLB.en-ig.en
6145395 NLLB.en-ig.ig
✅ Files aligned
```

---

## Step 2: Prepare Training Data

### Training Data Format

We create **4 training examples per sentence pair** to help the model learn bidirectional translation:

1. **English → Igbo translation**
2. **Igbo → English translation**
3. **English → Igbo with Igbo instruction**
4. **Igbo → English with Igbo instruction**

**Total training examples:** 6,145,395 × 4 = **19,471,872**

### Preparation Script

see 'scripts/prepare_nllb_training.py`

### Run the Preparation Script

```bash
# Make script executable
chmod +x scripts/prepare_nllb_training.py

# Run preparation
python scripts/prepare_nllb_training.py \
    --en-file ~/igbo/en-ig/NLLB.en-ig.en \
    --ig-file ~/igbo/en-ig/NLLB.en-ig.ig \
    --output-dir ~/igbo/datasets/processed-nllb
```

**Expected output:**
```
Reading ~/igbo/en-ig/NLLB.en-ig.en...
Reading ~/igbo/en-ig/NLLB.en-ig.ig...
Processing 6,145,395 sentence pairs...
Creating examples: 100%|████████| 6145395/6145395 [02:15<00:00, 45234.56it/s]
Created 24,581,580 training examples

Writing training file: ~/igbo/datasets/processed-nllb/igbo_train.jsonl
Writing train: 100%|████████| 24458672/24458672 [05:30<00:00, 74012.34it/s]
Writing validation file: ~/igbo/datasets/processed-nllb/igbo_val.jsonl
Writing val: 100%|████████| 122908/122908 [00:02<00:00, 73891.23it/s]

✅ Training data preparation complete!
   Train examples: 24,458,672
   Val examples: 122,908
   Train file: ~/igbo/datasets/processed-nllb/igbo_train.jsonl (5123.4 MB)
   Val file: ~/igbo/datasets/processed-nllb/igbo_val.jsonl (25.7 MB)
```

---

## Step 3: Verify Prepared Data

### Sample Training Examples

```bash
# View first 3 training examples
head -3 ~/igbo/datasets/processed-nllb/igbo_train.jsonl | python -m json.tool
```

**Example output:**
```json
{
  "instruction": "Translate to Igbo:",
  "input": "Good morning, how are you?",
  "output": "Ụtụtụ ọma, kedu ka ị mere?"
}
{
  "instruction": "Sụgharịa n'asụsụ Bekee (Translate to English):",
  "input": "Ọ bụ nnọọ mma ịhụ gị.",
  "output": "It's very nice to see you."
}
{
  "instruction": "Translate to English:",
  "input": "Kedu aha gị?",
  "output": "What is your name?"
}
```

### Verify File Sizes

```bash
# Check file sizes
ls -lh ~/igbo/datasets/processed-nllb/

# Count lines
wc -l ~/igbo/datasets/processed-nllb/*.jsonl
```

---

## Step 4: Upload to S3

### Create S3 Bucket (if not exists)

```bash
# Set your bucket name
BUCKET_NAME="learn-igbo-ekpes-useast1"

# Create bucket (in us-east-1 for training)
aws s3 mb s3://${BUCKET_NAME} --region us-east-1

# Enable versioning (optional but recommended)
aws s3api put-bucket-versioning \
    --bucket ${BUCKET_NAME} \
    --versioning-configuration Status=Enabled
```

### Upload Training Data

```bash
# Upload to S3
aws s3 sync ~/igbo/datasets/processed-nllb/ \
    s3://${BUCKET_NAME}/datasets/nllb/ \
    --exclude "*" \
    --include "*.jsonl"

# Verify upload
aws s3 ls s3://${BUCKET_NAME}/datasets/nllb/ --human-readable
```

**Expected output:**
```
2026-01-06 14:30:15    4.9 GiB igbo_train.jsonl
2026-01-06 14:30:45   24.5 MiB igbo_val.jsonl
```

### Create Train/Val Directory Structure for SageMaker

SageMaker expects separate directories for train and validation:

```bash
# Create separate directories
mkdir -p ~/igbo/datasets/nllb-sagemaker/train
mkdir -p ~/igbo/datasets/nllb-sagemaker/validation

# Copy files to appropriate directories
cp ~/igbo/datasets/processed-nllb/igbo_train.jsonl ~/igbo/datasets/nllb-sagemaker/train/
cp ~/igbo/datasets/processed-nllb/igbo_val.jsonl ~/igbo/datasets/nllb-sagemaker/validation/

# Upload to S3 with directory structure
aws s3 sync ~/igbo/datasets/nllb-sagemaker/ \
    s3://${BUCKET_NAME}/datasets/nllb/ \
    --delete

# Verify structure
aws s3 ls s3://${BUCKET_NAME}/datasets/nllb/ --recursive --human-readable
```

**Expected structure:**
```
s3://learn-igbo-ekpes-useast1/datasets/nllb/
├── train/
│   └── igbo_train.jsonl (4.9 GiB)
└── validation/
    └── igbo_val.jsonl (24.5 MiB)
```

---

## Storage Costs

### S3 Standard Storage

| Item | Size | Monthly Cost |
|------|------|--------------|
| Training data | 4.9 GB | $0.11 |
| Validation data | 24.5 MB | $0.00 |
| **Total** | **~5 GB** | **~$0.12/month** |

**Cost calculation:** $0.023 per GB/month in us-east-1

### S3 Data Transfer (One-time)

| Transfer | Size | Cost |
|----------|------|------|
| Upload to S3 | 5 GB | $0.00 (free) |
| Download by SageMaker | 5 GB | $0.00 (same region) |

**Note:** Transfers between S3 and SageMaker in the same region are free!

---

## Data Format Details

### Instruction Format

Each training example follows this structure:

```json
{
  "instruction": "Task description",
  "input": "Source text",
  "output": "Target text"
}
```

### Four Variations Per Sentence Pair

Given an English-Igbo pair:
- **EN:** "Hello, how are you?"
- **IG:** "Ndewo, kedu ka ị mere?"

We create 4 examples:

**1. English → Igbo (Simple)**
```json
{
  "instruction": "Translate to Igbo:",
  "input": "Hello, how are you?",
  "output": "Ndewo, kedu ka ị mere?"
}
```

**2. Igbo → English (Simple)**
```json
{
  "instruction": "Translate to English:",
  "input": "Ndewo, kedu ka ị mere?",
  "output": "Hello, how are you?"
}
```

**3. English → Igbo (Igbo Instruction)**
```json
{
  "instruction": "Sụgharịa n'asụsụ Igbo (Translate to Igbo):",
  "input": "Hello, how are you?",
  "output": "Ndewo, kedu ka ị mere?"
}
```

**4. Igbo → English (Igbo Instruction)**
```json
{
  "instruction": "Sụgharịa n'asụsụ Bekee (Translate to English):",
  "input": "Ndewo, kedu ka ị mere?",
  "output": "Hello, how are you?"
}
```

### Why 4 Variations?

1. **Bidirectional learning:** Model learns both EN→IG and IG→EN
2. **Instruction diversity:** Handles both English and Igbo instructions
3. **Robustness:** Better generalization to different prompting styles
4. **Cultural context:** Igbo speakers can use Igbo instructions

---

## Troubleshooting

### Issue: File encoding errors

```bash
# Check file encoding
file -i NLLB.en-ig.en
file -i NLLB.en-ig.ig

# Convert to UTF-8 if needed
iconv -f ISO-8859-1 -t UTF-8 NLLB.en-ig.en > NLLB.en-ig.en.utf8
```


### Issue: S3 upload slow

```bash
# Use multipart upload for large files
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB

# Upload with progress
aws s3 cp igbo_train.jsonl s3://bucket/path/ --storage-class STANDARD
```

---

## Next Steps

Once data is uploaded to S3, proceed to:
- **[03-sagemaker-training.md](03-sagemaker-training.md)** - Launch training on SageMaker
- Verify S3 paths are correct
- Ensure IAM role has S3 read permissions

---

## Quick Reference

```bash
# Download NLLB
wget https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-ig.txt.zip
unzip en-ig.txt.zip

# Prepare data
python scripts/prepare_nllb_training.py \
    --en-file NLLB.en-ig.en \
    --ig-file NLLB.en-ig.ig \
    --output-dir processed-nllb

# Upload to S3
aws s3 sync processed-nllb/ s3://your-bucket/datasets/nllb/
```

---

**Data preparation complete!** Ready for training. 🚀
