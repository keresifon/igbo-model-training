#!/usr/bin/env python3
"""
SageMaker training script for fine-tuning Mistral-7B on Igbo language data
FIXED VERSION - Proper checkpoint configuration to prevent data loss
"""

import os
import json
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CheckpointValidationCallback(TrainerCallback):
    """Callback to validate checkpoint saving and log checkpoint paths"""
    
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        logger.info(f"[CHECK] CHECKPOINT SAVED: {checkpoint_dir}")
        logger.info(f"   Step: {state.global_step}")
        logger.info(f"   Epoch: {state.epoch:.2f}")
        
        # Verify checkpoint files exist
        required_files = ['adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json']
        for file in required_files:
            file_path = os.path.join(checkpoint_dir, file)
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"   [OK] {file}: {size_mb:.2f} MB")
            else:
                logger.warning(f"   [WARNING]  Missing: {file}")
        
        return control


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    return parser.parse_args()


def format_instruction(example):
    if example.get('input'):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {'text': prompt}


def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("IGBO LANGUAGE MODEL TRAINING - FIXED CHECKPOINT VERSION")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_data_dir}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Train data: {args.train}")
    logger.info(f"Validation data: {args.validation}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info(f"Loading datasets from {args.train}")
    logger.info(f"Files in train directory: {os.listdir(args.train)}")
    
    train_dataset = load_dataset('json', data_files=f'{args.train}/*.jsonl', split='train')
    val_dataset = load_dataset('json', data_files=f'{args.validation}/*.jsonl', split='train')
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Format and tokenize
    logger.info("Formatting datasets...")
    train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_instruction, remove_columns=val_dataset.column_names)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.max_length, padding='max_length')
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # FIXED: Training arguments with proper checkpoint configuration
    logger.info("Configuring training arguments...")
    training_args = TrainingArguments(
        # CRITICAL: Use /opt/ml/output/data which auto-syncs to S3
        output_dir=args.output_data_dir,
        
        # Training params
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        
        # Optimization
        fp16=True,
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        
        # Logging
        logging_steps=100,
        logging_first_step=True,
        report_to="none",
        
        # Evaluation
        evaluation_strategy="steps",
        eval_steps=500,
        
        # FIXED: Checkpoint strategy - KEEP ALL CHECKPOINTS
        save_strategy="steps",
        save_steps=5000,  # Save every 5000 steps (more reasonable)
        save_total_limit=None,  # CRITICAL: Don't delete old checkpoints!
        
        # Don't try to load best model if job times out
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
    )
    
    logger.info("Training configuration:")
    logger.info(f"  Output dir: {training_args.output_dir}")
    logger.info(f"  Save strategy: {training_args.save_strategy}")
    logger.info(f"  Save steps: {training_args.save_steps}")
    logger.info(f"  Save total limit: {training_args.save_total_limit}")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    
    # Create trainer with checkpoint validation callback
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[CheckpointValidationCallback()],  # Validate checkpoints
    )
    
    # Start training
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    logger.info(f"Total training steps: {trainer.state.max_steps if trainer.state.max_steps else 'calculated at runtime'}")
    logger.info(f"Checkpoints will be saved every {training_args.save_steps} steps")
    logger.info(f"Checkpoints location: {training_args.output_dir}")
    
    try:
        trainer.train()
        logger.info("[CHECK] Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("[WARNING]  Training interrupted by user")
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed with error: {e}")
        raise
    
    finally:
        # ALWAYS save final model state, even if training was interrupted
        logger.info("=" * 80)
        logger.info("SAVING FINAL MODEL")
        logger.info("=" * 80)
        
        try:
            # Save to /opt/ml/model (final model location)
            logger.info(f"Saving to {args.model_dir}...")
            trainer.save_model(args.model_dir)
            tokenizer.save_pretrained(args.model_dir)
            logger.info("[CHECK] Saved to /opt/ml/model")
            
            # ALSO save to output_data_dir as backup
            final_output = os.path.join(args.output_data_dir, "final_model")
            logger.info(f"Saving backup to {final_output}...")
            trainer.save_model(final_output)
            tokenizer.save_pretrained(final_output)
            logger.info("[CHECK] Saved backup to output_data_dir/final_model")
            
            # Save training metrics
            metrics_file = os.path.join(args.output_data_dir, 'training_metrics.json')
            logger.info(f"Saving training metrics to {metrics_file}...")
            with open(metrics_file, 'w') as f:
                json.dump(trainer.state.log_history, f, indent=2)
            logger.info("[CHECK] Training metrics saved")
            
            # List all checkpoints
            logger.info("\n" + "=" * 80)
            logger.info("SAVED CHECKPOINTS:")
            logger.info("=" * 80)
            checkpoints = [d for d in os.listdir(args.output_data_dir) if d.startswith('checkpoint-')]
            checkpoints.sort(key=lambda x: int(x.split('-')[1]))
            for checkpoint in checkpoints:
                checkpoint_path = os.path.join(args.output_data_dir, checkpoint)
                size_mb = sum(os.path.getsize(os.path.join(checkpoint_path, f)) 
                             for f in os.listdir(checkpoint_path)) / (1024 * 1024)
                logger.info(f"  [OK] {checkpoint} ({size_mb:.2f} MB)")
            logger.info(f"\nTotal checkpoints: {len(checkpoints)}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving final model: {e}")
            raise
    
    logger.info("=" * 80)
    logger.info("TRAINING JOB COMPLETE!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()