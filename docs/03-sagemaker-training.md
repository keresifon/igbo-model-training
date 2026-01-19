# SageMaker Training Guide

This guide covers launching and managing the training job on AWS SageMaker.

## Overview

We'll train Mistral-7B-v0.1 using LoRA (Low-Rank Adaptation) on AWS SageMaker with spot instances for cost optimization.

**Training Configuration:**
- Model: Mistral-7B-v0.1 (7 billion parameters)
- Method: LoRA fine-tuning (trains ~40M parameters)
- Instance: ml.g5.xlarge (NVIDIA A10G, 24GB VRAM)
- Duration: 165-200 hours (~7-8 days)


---

## Prerequisites

Before starting, ensure you have completed:
- ✅ [01-setup.md](01-setup.md) - AWS account, quotas, IAM roles
- ✅ [02-data-preparation.md](02-data-preparation.md) - Data uploaded to S3

### Required S3 Structure

```
s3://your-bucket/
├── datasets/
│   └── nllb/
│       ├── train/
│       │   └── igbo_train.jsonl (4.9 GB)
│       └── validation/
│           └── igbo_val.jsonl (24.5 MB)
└── scripts/
    └── train_igbo_model.py
```

---

## Step 1: Prepare Training Script

### Create Training Script

See `scripts/train_igbo_model.py`:



## Step 2: Launch Training from SageMaker Studio

### Using SageMaker Studio (What You Used)

**📓 Complete Reference:** The full working notebook is available at [`scripts/igbo-train.ipynb`](../scripts/igbo-train.ipynb) in this repository.

**1. Open SageMaker Studio:**
   - Go to: https://console.aws.amazon.com/sagemaker/
   - Click "Studio" in the left menu
   - Click "Open Studio" for your user profile

**2. Create a New Notebook:**
   - In Studio, click File → New → Notebook
   - Select kernel: **Python 3 (Data Science 3.0)**
   - Name it: `igbo-train.ipynb`
   - Wait for kernel to start

   **Reference:** See the complete notebook in `scripts/igbo-train-ondemand.ipynb` in this repository.

**3. Upload Training Script:**
   - In Studio file browser (left sidebar)
   - Click the upload button (⬆️) at the top
   - Select `train_igbo_model.py` from your computer
   - The file will appear in your Studio environment
   - You can see it in the file browser

**4. Create Training Notebook (6 Cells):**

The complete working notebook is available at `scripts/igbo-train.ipynb`. 

---

**What Happens When You Run These Cells:**

1. **Cells 1-3:** Set up configuration
2. **Cell 4:** Create estimator (defines how to train)
3. **Cell 5:** Launch training job
   - SageMaker packages `train_igbo_model.py` from Studio
   - Uploads it to temporary S3 location
   - Provisions ml.g5.xlarge instance (or waits for spot)
   - Downloads script and data to instance
   - Starts training
4. **Cell 6:** Get monitoring link to track progress

**Important Notes:**
- You don't need to upload the script to S3 manually
- SageMaker handles the upload automatically when you run `estimator.fit()`
- The script is taken from your Studio environment
- Training runs independently (you can close the notebook)

---

**5. Verify Launch:**
   - Check SageMaker Console
   - Job should appear in "Training jobs"
   - Status will show "Starting" (waiting for spot instance)
   - Once instance is available: "Downloading" → "Training"

---


### CloudWatch Logs (Detailed View)

**Access directly:**
1. Go to: https://console.aws.amazon.com/cloudwatch/
2. Click "Logs" → "Log groups"
3. Find `/aws/sagemaker/TrainingJobs`
4. Click on your job's log stream
5. Click "Search log group" to filter

**Useful searches:**
- Search for: `loss` - See training loss values
- Search for: `ERROR` - Find any errors
- Search for: `Saving` - See checkpoint saves
- Search for: `Step` - Track training progress

---



## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```python
'per_device_train_batch_size': 1,  # Already at minimum
```

2. **Reduce sequence length:**
```python
'max_length': 128,  # From 256
```

3. **Enable gradient checkpointing:** (already enabled)
```python
gradient_checkpointing=True
```




### Issue: Training Stuck/Slow

**Check:**
1. CloudWatch logs for errors
2. Instance metrics (CPU/GPU utilization)
3. S3 data transfer speed
4. Network connectivity

**Solutions:**
```bash
# Check GPU utilization in logs
grep "GPU" /aws/sagemaker/TrainingJobs/job-name

# Verify data accessibility
aws s3 ls s3://your-bucket/datasets/nllb/train/
```

### Issue: Job Fails Immediately

**Common causes:**
1. IAM role missing S3 permissions
2. Training script syntax error
3. Incorrect S3 paths
4. Wrong library versions

**Debug:**
```bash
# Check failure reason
aws sagemaker describe-training-job --training-job-name JOB_NAME \
    | jq -r '.FailureReason'

# View full logs
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name JOB_NAME/algo-1-TIMESTAMP
```

---

## Post-Training

### Download Trained Model

```bash
# Model is automatically saved to S3
aws s3 ls s3://your-bucket/models/igbo-llm/

# Download
aws s3 cp s3://your-bucket/models/igbo-llm/JOB_NAME/output/model.tar.gz .

# Extract
tar -xzf model.tar.gz
```

### Model Artifacts

```
model.tar.gz contains:
├── adapter_config.json  # LoRA configuration
├── adapter_model.bin    # LoRA weights (~80MB)
├── tokenizer_config.json
├── tokenizer.json
└── special_tokens_map.json
```

---

## Next Steps

Once training completes:
- **[04-monitoring.md](04-monitoring.md)** - Analyze training metrics
- **[05-deployment.md](05-deployment.md)** - Deploy for inference
- Test the model with Igbo translations

---

