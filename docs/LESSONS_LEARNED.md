```markdown
# Production ML Training: Lessons Learned
## Igbo Language Model - January 2026

---

## 🎯 Overview

This document captures critical lessons from a 168-hour (7-day) production ML training run on AWS SageMaker. These insights are valuable for anyone running large-scale model training and can prevent costly mistakes.

---

## 📊 Training Summary

**Project:** Fine-tuning Mistral-7B (7B parameters) for Igbo-English translation  
**Duration:** 168 hours continuous  
**Cost:** $237  
**Infrastructure:** AWS SageMaker ml.g5.xlarge  
**Outcome:** Successfully completed with valuable production experience  

---

## 🔴 CRITICAL ISSUE: Checkpoint Configuration

### The Problem

**Initial Training Run:**
- Executed for 168 hours (completed full duration)
- Training appeared successful in CloudWatch logs
- GPU utilization maintained at 90-95%
- Loss convergence looked good
- **BUT:** Only checkpoint-1000 was preserved (out of ~30 expected checkpoints)

**Impact:**
- Lost 168 hours of training progress
- Lost $237 in compute costs
- Only had a barely-trained model (0.66% complete)
- Had to relaunch entire training job

### Root Cause Analysis

**Configuration Issue:**
```python
# PROBLEMATIC CONFIGURATION
training_args = TrainingArguments(
    output_dir="/opt/ml/output/data",  # Correct location
    save_steps=1000,
    save_total_limit=3,  # ❌ PROBLEM: Deletes old checkpoints!
)

# Without checkpoint_s3_uri, checkpoints saved locally
# With save_total_limit=3, only keeps last 3 checkpoints
# When job times out, only checkpoint-1000 remained