# AWS Setup Guide

This guide walks you through setting up your AWS account for SageMaker training.

## Step 1: Request SageMaker Quotas

New AWS accounts have 0 quota for GPU instances. You must request increases.

### Required Quotas

| Quota | Value | Approval Time |
|-------|-------|---------------|
| ml.g5.xlarge for training | 1 instance | 24-48 hours |
| Longest run time for training | 604800 seconds | Instant |

### Request Process

1. **Go to Service Quotas:**
   https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas

2. **Select Region:** us-east-1 (top right)

3. **Search for quota:** `ml.g5.xlarge for training job usage`

4. **Click on it**

5. **Click "Request increase at account-level"**

6. **Enter:**
   - New quota value: `1`
   - Use case description: "Training Igbo language model for educational purposes using Llama/Mistral with LoRA fine-tuning on NLLB dataset (6M sentence pairs)"

7. **Submit request**



### Check Approval Status

```bash
aws service-quotas get-service-quota \
  --service-code sagemaker \
  --quota-code L-9828A5EF \
  --region us-east-1
```

Or check email for approval notification.

## Step 2: Set Up IAM Role for SageMaker

SageMaker needs permissions to access S3 and other AWS services.

### Create Execution Role

1. **Go to IAM Console:**
   https://console.aws.amazon.com/iam/home#/roles

2. **Click "Create role"**

3. **Select:**
   - Trusted entity: AWS service
   - Use case: SageMaker

4. **Attach policy:**
   - `AmazonSageMakerFullAccess` (automatically attached)

5. **Name:** `SageMaker-IgboTraining-Role`

6. **Create role**

7. **Add S3 permissions with inline policy:**
   - Click on the newly created role
   - Click "Add permissions" → "Create inline policy"
   - Click "JSON" tab
   - Paste this policy (replace `your-bucket-name` with your actual bucket):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3BucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    },
    {
      "Sid": "S3BucketList",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": "arn:aws:s3:::your-bucket-name"
    }
  ]
}
```

   - Name the policy: `S3IgboTrainingAccess`
   - Click "Create policy"

8. **Copy ARN** (you'll need this later)

**Security note:** This grants access only to your specific bucket, following the principle of least privilege. Avoid using `AmazonS3FullAccess` which grants access to all S3 buckets in your account.

---

## Step 3: Create S3 Bucket

**Using S3 Console:**

1. **Go to S3 Console:**
   https://console.aws.amazon.com/s3/

2. **Click "Create bucket"**

3. **Configure bucket:**
   - **Bucket name:** `learn-igbo-ekpes-useast1` (must be globally unique)
   - **Region:** US East (N. Virginia) us-east-1
   - **Block Public Access:** Keep all enabled (recommended)
   - **Bucket Versioning:** Enable (optional, for safety)

4. **Click "Create bucket"**

**Naming requirements:**
- Globally unique
- 3-63 characters
- Lowercase letters, numbers, hyphens
- No underscores or spaces


---

## Step 4: Set Up SageMaker Domain (For Studio)

If using SageMaker Studio notebooks:

### Quick Setup

1. **Go to SageMaker Console:**
   https://console.aws.amazon.com/sagemaker/home

2. **Click "Domains"** (left sidebar)

3. **Click "Create domain"**

4. **Select "Quick setup"**

5. **Configure:**
   - Domain name: `igbo-training-domain`
   - Execution role: Create new role with S3 access

6. **Click "Submit"**

7. **Wait 5-10 minutes** for domain creation

### Launch Studio

1. **Click domain name**
2. **Click "Launch"** → "Studio"
3. **Wait 2-3 minutes** for Studio to load

---

## Step 5: Verify Setup

**Check in AWS Console:**

1. **IAM Role:**
   - Go to: https://console.aws.amazon.com/iam/home#/roles
   - Find `SageMaker-IgboTraining-Role`
   - Verify it has AmazonSageMakerFullAccess policy
   - Verify it has S3IgboTrainingAccess inline policy

2. **S3 Bucket:**
   - Go to: https://console.aws.amazon.com/s3/
   - Find your bucket (e.g., `learn-igbo-ekpes-useast1`)
   - Verify region is us-east-1

3. **SageMaker Domain:**
   - Go to: https://console.aws.amazon.com/sagemaker/
   - Click "Domains" in left menu
   - Verify domain status shows "InService"
   - Verify user profile exists

4. **Service Quotas:**
   - Go to: https://console.aws.amazon.com/servicequotas/
   - Search for "ml.g5.xlarge for spot training"
   - Verify quota value = 1 (or request is approved)

---

## Common Issues

### Issue: Quota Request Still Pending

**Problem:** Request still under review

**Solution:** Wait 24-48 hours, check email for updates

### Issue: SageMaker Domain Won't Create

**Problem:** IAM permissions insufficient

**Solution:** Ensure execution role has AmazonSageMakerFullAccess

---



## Next Steps

Once your quotas are approved (24-48 hours):

1. [Prepare training data](02-data-preparation.md)
2. [Launch SageMaker training](03-sagemaker-training.md)

## Support

- AWS Support: https://aws.amazon.com/support/
- AWS Forums: https://forums.aws.amazon.com/
- AWS Documentation: https://docs.aws.amazon.com/

---

