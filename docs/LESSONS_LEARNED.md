# Production ML Training: Lessons Learned
## Igbo Language Model - January 2026

---

## 🎯 Overview

This document captures critical lessons from a 168-hour (7-day) production ML training run on AWS SageMaker. These insights are valuable for anyone running large-scale model training and can prevent costly mistakes.

**Project:** Fine-tuning Mistral-7B (7B parameters) for Igbo-English translation  
**Duration:** 168 hours continuous  
**Cost:** $237  
**Infrastructure:** AWS SageMaker ml.g5.xlarge  
**Outcome:** Successfully completed with valuable production experience  

---

## 📊 Training Summary

### Configuration
- **Base Model:** Mistral-7B-v0.1 (7 billion parameters)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Trainable Parameters:** 6.8M (0.094% of base model)
- **Dataset:** 6.1M Igbo-English sentence pairs (19.5M training examples)
- **Instance:** ml.g5.xlarge (NVIDIA A10G, 24GB VRAM)
- **Duration:** 168 hours (7 days continuous)

### Results
- **GPU Utilization:** 90-95% maintained throughout
- **Peak Memory:** 23GB / 24GB VRAM
- **Throughput:** 11.56 iterations/second
- **Checkpoints:** ~30 saved (every 5000 steps)
- **OOM Errors:** 0 (zero)
- **Final Cost:** $237 (on-demand pricing)

---

## 🔴 CRITICAL LESSON #1: Checkpoint Configuration

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
- Only had a barely-trained model (0.66% complete at step 1000)
- Had to completely relaunch entire training job

### Root Cause Analysis

**The Problematic Configuration:**
```python
# WHAT WE HAD (BROKEN)
training_args = TrainingArguments(
    output_dir="/opt/ml/output/data",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,  # ❌ PROBLEM: Deletes old checkpoints!
    load_best_model_at_end=True,
)
```

**What Actually Happened:**
1. **Step 1000:** checkpoint-1000 saved ✅
2. **Step 2000:** checkpoint-2000 saved ✅
3. **Step 3000:** checkpoint-3000 saved ✅
4. **Step 4000:** checkpoint-4000 saved → checkpoint-1000 **DELETED** (keeping only last 3) ❌
5. **Step 5000:** checkpoint-5000 saved → checkpoint-2000 **DELETED** ❌
6. This continued for 168 hours...
7. **Step 152000:** Job hit 7-day timeout before next checkpoint at step 153000
8. SageMaker uploaded /opt/ml/output/data/ contents to S3
9. **Result:** Only checkpoint-1000 remained (all others had been deleted during training)

**Why It Happened:**
- `save_total_limit=3` means "keep only the last 3 checkpoints"
- During training, older checkpoints are automatically deleted
- When job times out, only the remaining checkpoints get uploaded
- In our case, only checkpoint-1000 survived the deletion cycle

### The Fix

**CORRECTED CONFIGURATION:**
```python
# WHAT WE NEED (WORKING)
training_args = TrainingArguments(
    output_dir="/opt/ml/output/data",
    save_strategy="steps",
    save_steps=5000,  # Less frequent = more efficient
    save_total_limit=None,  # ✅ CRITICAL: Keep ALL checkpoints!
    load_best_model_at_end=False,  # Can't load if job times out
)
```

**Additional Safety Measures Implemented:**

1. **Checkpoint Validation Callback:**
```python
from transformers import TrainerCallback
import logging
import os

class CheckpointValidationCallback(TrainerCallback):
    """Validates checkpoint saving and logs details"""
    
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(
            args.output_dir, 
            f"checkpoint-{state.global_step}"
        )
        
        logger.info(f"[CHECK] CHECKPOINT SAVED: {checkpoint_dir}")
        logger.info(f"   Step: {state.global_step}")
        logger.info(f"   Epoch: {state.epoch:.2f}")
        
        # Verify files exist
        required_files = [
            'adapter_model.safetensors',
            'adapter_config.json',
            'trainer_state.json'
        ]
        
        for file in required_files:
            file_path = os.path.join(checkpoint_dir, file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"   [OK] {file}: {size_mb:.2f} MB")
            else:
                logger.warning(f"   [WARNING] Missing: {file}")
        
        return control
```

2. **Backup Model Save in Finally Block:**
```python
try:
    trainer.train()
    logger.info("Training completed successfully!")
    
except Exception as e:
    logger.error(f"Training failed with error: {e}")
    raise
    
finally:
    # ALWAYS save final model, even if training interrupted
    logger.info("Saving final model...")
    
    # Save to /opt/ml/model (final location)
    trainer.save_model("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")
    
    # ALSO save to output_data_dir (backup)
    final_output = os.path.join(args.output_data_dir, "final_model")
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    
    logger.info("Final model saved to both locations")
```

3. **List All Checkpoints at End:**
```python
# Log all saved checkpoints
checkpoints = [d for d in os.listdir(args.output_data_dir) 
               if d.startswith('checkpoint-')]
checkpoints.sort(key=lambda x: int(x.split('-')[1]))

logger.info("\n" + "=" * 80)
logger.info("SAVED CHECKPOINTS:")
logger.info("=" * 80)
for checkpoint in checkpoints:
    checkpoint_path = os.path.join(args.output_data_dir, checkpoint)
    size_mb = sum(os.path.getsize(os.path.join(checkpoint_path, f)) 
                 for f in os.listdir(checkpoint_path)) / (1024 * 1024)
    logger.info(f"  [OK] {checkpoint} ({size_mb:.2f} MB)")
logger.info(f"\nTotal checkpoints preserved: {len(checkpoints)}")
```

### Prevention Checklist

Before starting any long-running training job:

- [ ] Set `save_total_limit=None` (or sufficiently high number like 100)
- [ ] Choose appropriate `save_steps` frequency (5000 for 7-day jobs)
- [ ] Add checkpoint validation callback to training script
- [ ] Test checkpoint saving in first 30-60 minutes
- [ ] Verify checkpoint files appear in output directory
- [ ] Check S3 for uploaded checkpoints (if using checkpoint_s3_uri)
- [ ] Implement backup save locations (finally block)
- [ ] Add checkpoint counting/listing at end of training

### Key Takeaway

**Never use `save_total_limit` with a low value for critical long-running training jobs.**

The "optimization" of keeping only N checkpoints can backfire catastrophically if:
- Training times out before reaching a checkpoint boundary
- You need to debug from an earlier training state
- You want to compare model quality at different training stages

**For production training:** Disk space is cheap, training time is expensive. Keep all checkpoints.

---

## 💰 CRITICAL LESSON #2: Cost vs Reliability Tradeoff

### The Problem

**Initial Approach: Spot Instances**
- **Hourly Cost:** $0.42/hour (70% discount vs on-demand)
- **Expected 7-day Cost:** $71 (if uninterrupted)
- **Reality:** Interrupted after 41 hours due to capacity reclamation

**What Happened:**
1. Launched training on spot instance
2. Training started successfully
3. After 41 hours: Spot capacity interruption
4. Job terminated, checkpoint uploaded
5. Attempted to resume → No spot capacity available
6. Waited 2 hours → Still no capacity
7. **Decision:** Switch to on-demand

### The Decision

**Switched to On-Demand Instances:**
- **Hourly Cost:** $1.41/hour
- **Total 7-day Cost:** $237
- **Result:** Zero interruptions, completed successfully

### Cost-Benefit Analysis

| Factor | Spot Instances | On-Demand |
|--------|---------------|-----------|
| **Hourly Rate** | $0.42 | $1.41 |
| **7-Day Cost** | $71 (theoretical) | $237 (actual) |
| **Reliability** | Interrupted after 41hrs | 100% uptime |
| **Risk** | May lose progress | Guaranteed completion |
| **Total Time** | Unknown (interruptions) | 168 hours (7 days) |
| **Peace of Mind** | Low | High |

**Actual Costs:**
- Spot attempt: $17 (41 hours before interruption)
- On-demand run: $237 (168 hours, completed)
- **Total spent:** $254
- **vs Pure spot (if worked):** $71
- **Premium paid:** $183 for guaranteed completion

### The Learning

**For 7-Day Training Jobs:**
- Spot instances are **high risk** for interruptions
- Each interruption requires manual intervention
- Multiple interruptions can **double total time**
- Time value of completing 7 days earlier matters
- **Recommendation:** Use on-demand for jobs >3 days

### When to Use Spot vs On-Demand

**Good for Spot Instances:**
- ✅ Training jobs <24 hours
- ✅ Experimentation and testing
- ✅ Jobs with frequent checkpoints (every 1000 steps)
- ✅ Not time-sensitive work
- ✅ Multiple parallel experiments
- ✅ Easy to restart/resume

**Use On-Demand Instead:**
- ✅ Production training >3 days
- ✅ Time-sensitive deadlines
- ✅ Final model training
- ✅ When reliability matters more than cost
- ✅ Jobs requiring 100% uptime
- ✅ High-value training runs

### Decision Framework

**Calculate the "interruption premium":**
```
On-demand premium = (On-demand rate - Spot rate) × Training hours
Time value = Days saved × Value of your time
Risk cost = Probability of failure × Cost of failure

If (Time value + Risk cost) > On-demand premium:
    Use on-demand
Else:
    Use spot
```

**For our case:**
- On-demand premium: ($1.41 - $0.42) × 168 = $166
- Time value: 7+ days earlier = High
- Risk cost: 50% chance of >1 interruption × $237 wasted = $119
- **Decision:** $166 < ($119 + time value) → Use on-demand ✅

### Key Takeaway

**Spot instances are not always cheaper in practice.**

For long-running training jobs, the hidden costs of interruptions (time, frustration, potential lost progress) often exceed the nominal savings. **Optimize for reliability first, then cost.**

---

## 🏗️ CRITICAL LESSON #3: Infrastructure Planning

### The Challenge

**AWS Service Quotas:**
- **Default limit:** 5 days (120 hours) max training duration
- **Our need:** 7 days (168 hours)
- **Risk:** Training would auto-stop at 5 days without extension

### The Problem

If we hadn't checked quotas beforehand:
1. Training would start successfully
2. Run for 5 days (120 hours)
3. Automatically terminate at quota limit
4. Lose 5 days of progress
5. Discover quota limit only after failure

### The Solution

**Proactive Quota Request:**
1. **Before training:** Checked Service Quotas in AWS Console
2. **Identified constraint:** Max training job duration = 5 days
3. **Submitted request:** Requested extension to 7 days
4. **Provided justification:** ML model training requirements
5. **Approval time:** 24 hours
6. **Result:** Extended from 5 days → 7 days

### How to Request Service Quota Increase

```bash
# Via AWS Console
1. Go to Service Quotas console
2. Search: "SageMaker"
3. Find: "Max training job duration"
4. Click "Request quota increase"
5. New value: 604800 seconds (7 days)
6. Business justification: "Training large language model"
7. Submit request
8. Wait for approval email (typically 24-48 hours)

# Via AWS CLI
aws service-quotas request-service-quota-increase \
    --service-code sagemaker \
    --quota-code L-XXXXXXXX \
    --desired-value 604800 \
    --region us-east-1
```

### The Learning

**Always Check Constraints Before Training:**

**Critical AWS Quotas for ML Training:**
- ✅ Max training job duration (hours)
- ✅ Instance type availability (ml.g5.xlarge, etc.)
- ✅ Number of concurrent training jobs
- ✅ VPC limits (if using VPC mode)
- ✅ S3 request limits
- ✅ CloudWatch log retention

**Timeline for Planning:**
- Request quotas **at least 48 hours** before training
- Some quotas are auto-approved (instant)
- Others require manual review (1-3 business days)
- Complex requests may need AWS support ticket

### Infrastructure Checklist

Before launching production training:

- [ ] Check instance type quota (ml.g5.xlarge availability)
- [ ] Verify max training duration quota
- [ ] Confirm S3 bucket limits and permissions
- [ ] Test IAM role has all required permissions
- [ ] Verify CloudWatch log group created
- [ ] Check network connectivity (if using VPC)
- [ ] Confirm spot instance limits (if using spot)
- [ ] Validate training data accessible from SageMaker

### Key Takeaway

**Infrastructure planning prevents painful surprises.**

Spending 30 minutes checking quotas beforehand can prevent losing days of training time. **Always validate constraints before starting expensive compute jobs.**

---

## 📈 LESSON #4: Monitoring & Validation

### What Worked Well

**CloudWatch Monitoring:**
- ✅ Real-time training logs
- ✅ GPU utilization tracking (maintained 90-95%)
- ✅ Loss convergence visualization
- ✅ Error detection and alerting
- ✅ Cost tracking

**Key Metrics We Tracked:**
```python
# Logged every 100 steps
- Training loss
- Learning rate
- Gradient norm
- GPU utilization

# Logged every 500 steps
- Validation loss
- Evaluation metrics

# Logged every 5000 steps
- Checkpoint saves
- Model size
- Training time elapsed
```

### What We Learned

**Early Validation is Critical:**

Don't wait 7 days to discover configuration issues!

**Validation Timeline:**
```
T+0 (Launch):
    ✅ Job starts, status = "InProgress"
    ✅ CloudWatch log stream created

T+30 minutes:
    ✅ First training logs appear
    ✅ Verify: "STARTING TRAINING"
    ✅ Check: GPU detected, model loaded
    ❌ If stuck: Check S3 data access

T+1 hour:
    ✅ Training loss decreasing
    ✅ First checkpoint saved (step 1000-5000)
    ✅ Verify checkpoint files exist locally
    ❌ If no checkpoint: Fix save configuration NOW

T+2 hours:
    ✅ Second checkpoint saved
    ✅ Confirm checkpoints NOT being deleted
    ✅ Verify save_total_limit=None working
    ❌ If checkpoints disappearing: STOP and fix

T+12 hours:
    ✅ Multiple checkpoints accumulated
    ✅ Loss convergence steady
    ✅ GPU utilization stable 90-95%
    ✅ No memory warnings

T+24 hours:
    ✅ ~15% complete (1 epoch done)
    ✅ Validation loss improving
    ✅ No OOM errors
    ✅ Checkpoints every 5000 steps
```

**The Critical 2-Hour Window:**

The first 2 hours of training are the most important for validation. This is when you can catch:
- ✅ Checkpoint configuration issues
- ✅ Memory problems (OOM)
- ✅ Data loading issues
- ✅ Permission errors
- ✅ Configuration mistakes

**If anything is wrong in the first 2 hours, STOP and fix it.** Don't waste 7 days on a misconfigured run.

### Monitoring Best Practices

**Set Up Alerts:**
```python
# CloudWatch Alarms to create:
1. GPU utilization <80% for >30 minutes
2. Training loss not decreasing for >2 hours
3. No new checkpoints in expected timeframe
4. Memory usage >95% of available
5. Error keywords in logs ("OOM", "CUDA error", "Failed")
```

**Create Monitoring Dashboard:**
```python
# Key widgets to include:
1. Training loss over time (line chart)
2. Validation loss over time (line chart)
3. GPU utilization (gauge)
4. Memory usage (gauge)
5. Training speed (iterations/sec)
6. Cost to date (calculated metric)
7. Estimated completion time
```

**Use CloudWatch Insights Queries:**
```sql
-- Find all checkpoint saves
fields @timestamp, @message
| filter @message like /CHECKPOINT SAVED/
| sort @timestamp desc

-- Check for errors
fields @timestamp, @message
| filter @message like /ERROR|OOM|CUDA/
| sort @timestamp desc

-- Track training progress
fields @timestamp, @message
| filter @message like /{'loss':/
| parse @message /'loss': *, / as loss
| stats avg(loss) by bin(5m)
```

### Key Takeaway

**Validate early and often.**

The first 2 hours of training tell you if the next 7 days will succeed. Invest time in monitoring setup and early validation to avoid wasting compute costs.

---

## 🎓 LESSON #5: Memory Optimization

### The Challenge

Initial training attempt failed immediately with:
```
OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB
```

**Problem:** Mistral-7B + batch_size=2 + max_length=512 = ~27GB needed
**Hardware:** ml.g5.xlarge only has 24GB VRAM

### The Solution

**Memory Optimization Strategy:**

1. **Reduced Batch Size:**
```python
# Before: 
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
# Memory: ~14GB for activations

# After:
per_device_train_batch_size = 1  # Saves ~4GB
gradient_accumulation_steps = 16  # Maintains effective batch=16
# Memory: ~8GB for activations
```

2. **Reduced Sequence Length:**
```python
# Before:
max_length = 512  # Memory: ~8GB

# After:
max_length = 256  # Saves ~2GB, sufficient for 95% of Igbo sentences
# Memory: ~4GB
```

3. **Enabled Gradient Checkpointing:**
```python
training_args = TrainingArguments(
    gradient_checkpointing=True,  # Saves ~2-3GB
    gradient_checkpointing_kwargs={'use_reentrant': False},
)
```

4. **Reduced LoRA Target Modules:**
```python
# Before:
target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # 4 modules

# After:
target_modules=["q_proj", "v_proj"]  # 2 modules, saves ~1GB
# Still effective: 6.8M trainable parameters
```

### Results

**Memory Budget After Optimization:**
```
Component                  Before    After    Savings
────────────────────────────────────────────────────
Model (FP16)               14GB      14GB     -
LoRA adapters              1GB       0.2GB    0.8GB
Activations (batch)        8GB       2GB      6GB
Gradients                  3GB       2GB      1GB
Optimizer states           2GB       3GB      -1GB
Buffers & overhead         -         1GB      -
────────────────────────────────────────────────────
TOTAL                      28GB      23GB     5GB
Available VRAM             24GB      24GB
Headroom                   -4GB ❌   1GB ✅   
```

**Production Validation:**
- Peak memory: 23.1GB / 24GB (95.8% utilized)
- Stable throughout 168 hours
- Zero OOM errors
- 4% headroom maintained

### Key Takeaway

**Memory optimization is a multi-dimensional problem.**

No single change fixes everything. You need to:
- Reduce batch size (immediate big savings)
- Reduce sequence length (appropriate for your data)
- Enable gradient checkpointing (trades compute for memory)
- Optimize model architecture (fewer LoRA modules if appropriate)

**Test memory configuration in first hour, not after 7 days.**

---

## 💡 LESSON #6: Production vs Experimentation

### What We Learned

**Experimentation (spot instances, aggressive optimization):**
- ✅ Good for: Testing ideas, iterating quickly, low cost
- ❌ Bad for: Production training, time-sensitive work

**Production (on-demand, defensive configuration):**
- ✅ Good for: Final model training, guaranteed completion
- ❌ Bad for: Quick tests, budget-constrained experiments

### Production-Ready Checklist

Before calling something "production-ready":

**Configuration:**
- [ ] `save_total_limit=None` (keep all checkpoints)
- [ ] Appropriate `save_steps` frequency
- [ ] Checkpoint validation callbacks
- [ ] Backup save locations (finally block)
- [ ] Memory optimization validated
- [ ] On-demand instances for reliability

**Monitoring:**
- [ ] CloudWatch alarms configured
- [ ] Monitoring dashboard created
- [ ] Error alerts set up
- [ ] Cost tracking enabled
- [ ] Progress tracking automated

**Documentation:**
- [ ] Configuration documented
- [ ] Known issues recorded
- [ ] Recovery procedures written
- [ ] Cost estimates accurate
- [ ] Timeline expectations realistic

**Validation:**
- [ ] Tested checkpoint saving in first 2 hours
- [ ] Verified memory usage stable
- [ ] Confirmed GPU utilization high (>85%)
- [ ] Validated data pipeline works
- [ ] Tested recovery from interruption

### Key Takeaway

**Production is different from experimentation.**

What works for a 1-hour test may fail catastrophically in a 7-day run. Invest in:
- Defensive checkpoint strategies
- Comprehensive monitoring
- Early validation
- Clear documentation

---

## 📊 Summary of Key Lessons

### 1. Checkpoint Configuration (CRITICAL)
**Problem:** save_total_limit deleted old checkpoints  
**Cost:** $237 + 168 hours lost  
**Solution:** save_total_limit=None + validation callbacks  
**Prevention:** Test checkpoint saving in first 2 hours  

### 2. Spot vs On-Demand
**Problem:** Spot interruption after 41 hours  
**Cost:** $17 wasted + 2 days delay  
**Solution:** On-demand for 7-day jobs ($237)  
**Prevention:** Choose instance type based on job duration  

### 3. Infrastructure Planning
**Problem:** Nearly hit 5-day quota limit  
**Cost:** Would have lost 5 days of training  
**Solution:** Proactive quota request (approved in 24hrs)  
**Prevention:** Check all quotas before training  

### 4. Memory Optimization
**Problem:** CUDA OOM on first batch  
**Cost:** Failed immediately  
**Solution:** batch=1, max_len=256, gradient checkpointing  
**Prevention:** Test memory in first hour  

### 5. Monitoring & Validation
**Problem:** Didn't validate checkpoints early  
**Cost:** Discovered issue after 7 days  
**Solution:** Validate in first 2 hours  
**Prevention:** Early validation checklist  

### 6. Production Mindset
**Problem:** Treating production like experimentation  
**Cost:** Multiple failed attempts  
**Solution:** Defensive configuration, on-demand instances  
**Prevention:** Production-ready checklist  

---

## 💰 Total Cost of Learning

**Direct Costs:**
- First run (spot, 41 hours): $17
- Second run (on-demand, 168 hours with checkpoint issue): $237
- Third run (on-demand, 168 hours successful): $237
- **Total spent:** $491

**Time Costs:**
- First run: 41 hours + 2 days recovery
- Second run: 168 hours (7 days)
- Third run: 168 hours (7 days)
- **Total time:** 16+ days

**Value Gained:**
- ✅ Deep understanding of checkpoint strategies
- ✅ Experience with production ML failures and recovery
- ✅ Knowledge of AWS SageMaker limitations
- ✅ Real cost-reliability tradeoff analysis
- ✅ Production ML operations expertise
- ✅ Documented learnings to help others

**ROI:**
- Prevents future multi-thousand dollar mistakes
- Builds expertise valuable for large-scale ML
- Creates reusable templates and best practices
- Demonstrates problem-solving and resilience

**The investment in learning was expensive, but invaluable.**

---

## 🎯 Actionable Recommendations

### For Your Next Training Run

**Before Training:**
1. Set `save_total_limit=None` in TrainingArguments
2. Choose `save_steps` appropriately (5000 for 7-day jobs)
3. Add checkpoint validation callbacks
4. Check all AWS service quotas
5. Use on-demand for jobs >3 days
6. Test memory configuration locally if possible

**During First 2 Hours:**
1. Verify training started (CloudWatch logs)
2. Check first checkpoint saved
3. Confirm checkpoint NOT deleted
4. Monitor GPU utilization (>85%)
5. Watch for memory warnings
6. Validate loss decreasing

**During Training:**
1. Check progress daily
2. Monitor cost accumulation
3. Verify checkpoints accumulating
4. Watch for anomalies in logs
5. Keep alert on CloudWatch alarms

**After Training:**
1. Verify all checkpoints uploaded to S3
2. Download and test model
3. Document any issues encountered
4. Update cost estimates
5. Share learnings with team

---

## 📚 References

- **Training Script:** `scripts/train_igbo_model_FIXED.py`
- **Notebook:** `scripts/igbo-train.ipynb`
- **Memory Optimizations:** `MEMORY_OPTIMIZATION_CHANGES.md`
- **AWS Docs:** [SageMaker Checkpointing](https://docs.aws.amazon.com/sagemaker/latest/dg/model-checkpoints.html)
- **HuggingFace:** [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- **LoRA Paper:** [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

---

## ✅ Conclusion

These lessons represent real production ML engineering experience gained through trial, error, and perseverance. The checkpoint configuration issue, while costly, provided invaluable knowledge that:

- ✅ Prevents similar issues in future training runs
- ✅ Helps others avoid the same mistakes
- ✅ Demonstrates understanding of production ML operations
- ✅ Shows ability to troubleshoot, document, and recover from failures
- ✅ Builds expertise in cost-reliability tradeoffs
- ✅ Creates reusable best practices and templates
