# Monitoring and Metrics Guide

This guide covers monitoring your training job and analyzing the results.

## Overview

Monitor training progress through:
- SageMaker Console (visual interface)
- CloudWatch Logs (detailed logs)
- CloudWatch Metrics (GPU/CPU utilization)
- Training metrics (loss, learning rate)

---

## SageMaker Console Monitoring

### Access Training Job

1. Go to https://console.aws.amazon.com/sagemaker/
2. Navigate to **Training** → **Training jobs**
3. Find your job (sort by creation date)

### Job Details

**Status States:**
- `InProgress` - Job is running
  - `Starting` - Provisioning instance
  - `Downloading` - Downloading data and model
  - `Training` - Active training
- `Completed` - Successfully finished ✅
- `Failed` - Error occurred ❌
- `Stopped` - Manually terminated

**Key Metrics:**
- **Billable time:** Actual training time (excludes setup)
- **Training time:** Total time including setup
- **Instance type:** ml.g5.xlarge


### Cost Tracking

```
Current cost = (Billable seconds / 3600) × $1.41
```



---

## CloudWatch Logs

### View Logs in Console

1. Open CloudWatch Console
2. Go to **Logs** → **Log groups**
3. Find `/aws/sagemaker/TrainingJobs`
4. Select your job's log stream

### Log Stream Name Format

```
/aws/sagemaker/TrainingJobs
└── your-job-name/algo-1/1234567890123
```


### Important Log Patterns

**Successful Start:**
```
INFO:__main__:STARTING IGBO LANGUAGE MODEL TRAINING
INFO:__main__:✓ Model loaded: mistralai/Mistral-7B-v0.1
INFO:__main__:✓ LoRA adapters configured
trainable params: 41,943,040 || all params: 7,283,943,424
INFO:__main__:✓ Train examples: 24,458,672
INFO:__main__:✓ Val examples: 122,908
INFO:__main__:STARTING TRAINING
```

**Training Progress:**
```
{'loss': 2.456, 'learning_rate': 0.0002, 'epoch': 0.01}
{'loss': 2.123, 'learning_rate': 0.00019, 'epoch': 0.05}
{'loss': 1.876, 'learning_rate': 0.00018, 'epoch': 0.10}
```

**Checkpoint Saves:**
```
Saving model checkpoint to /opt/ml/output/checkpoint-1000
```

**Errors to Watch For:**
```
OutOfMemoryError: CUDA out of memory
FileNotFoundError: [Errno 2] No such file or directory
ConnectionError: Failed to download model
```

---

## CloudWatch Metrics

### Built-in SageMaker Metrics

**Navigate to CloudWatch Metrics:**
1. CloudWatch Console → Metrics → All metrics
2. Select **SageMaker** → **TrainingJobName**

**Available Metrics:**
- `CPUUtilization` - CPU usage (0-100%)
- `MemoryUtilization` - RAM usage (0-100%)
- `GPUUtilization` - GPU usage (0-100%)
- `GPUMemoryUtilization` - VRAM usage (0-100%)
- `DiskUtilization` - Disk usage (0-100%)

### Expected Metric Ranges

| Metric | Normal Range | Alert If |
|--------|--------------|----------|
| GPU Utilization | 80-100% | <50% (underutilized) |
| GPU Memory | 70-90% | >95% (OOM risk) |
| CPU Utilization | 20-40% | >80% (bottleneck) |
| Memory | 30-50% | >90% (RAM issue) |
| Disk | 40-60% | >90% (storage issue) |




### Job Keeps Failing

1. Check CloudWatch logs for exact error
2. Verify S3 data accessibility
3. Test training script locally first
4. Reduce batch size/sequence length

---

## Best Practices

### 1. Monitor Regularly

- First hour: Check every 15 minutes
- First day: Check 2-3 times
- After day 1: Check once daily

### 2. Save Checkpoints

- Keep at least 2 checkpoints
- Upload to S3 regularly
- Test checkpoint loading



## Next Steps

After training completes successfully:
- Download trained model from S3
- Proceed to deployment: **[05-deployment.md](05-deployment.md)**

---



**Happy monitoring!** 📊 Your model is training! 🚀
