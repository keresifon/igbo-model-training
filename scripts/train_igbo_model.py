#!/usr/bin/env python3
"""
SageMaker training script for fine-tuning LLM on Igbo language data
This script runs inside the SageMaker training container
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
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific parameters
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    # SageMaker environment variables
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    return parser.parse_args()

def format_instruction(example):
    """Format examples for instruction fine-tuning"""
    if example.get('input'):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    
    return {'text': prompt}

def main():
    args = parse_args()
    
    logger.info("="*70)
    logger.info("STARTING IGBO LANGUAGE MODEL TRAINING")
    logger.info("="*70)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Train data: {args.train}")
    logger.info(f"Val data: {args.validation}")
    logger.info(f"Output: {args.model_dir}")
    
    # Load tokenizer and model
    logger.info("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    logger.info(f"✓ Model loaded: {args.model_name}")
    
    # Configure LoRA for efficient fine-tuning
    logger.info("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # Reduced from 4 to 2 modules to save memory
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Note: prepare_model_for_kbit_training is only needed for quantized models (8-bit/4-bit)
    # We're using FP16, so we skip it and go directly to LoRA
    model = get_peft_model(model, lora_config)
    
    logger.info("✓ LoRA adapters configured")
    model.print_trainable_parameters()
    
    # Load datasets
    logger.info("\nLoading training data...")
    train_dataset = load_dataset(
        'json',
        data_files=f'{args.train}/igbo_train.jsonl',
        split='train'
    )
    
    val_dataset = load_dataset(
        'json',
        data_files=f'{args.validation}/igbo_val.jsonl',
        split='train'
    )
    
    logger.info(f"✓ Train examples: {len(train_dataset):,}")
    logger.info(f"✓ Val examples: {len(val_dataset):,}")
    
    # Format data
    logger.info("\nFormatting data...")
    train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_instruction, remove_columns=val_dataset.column_names)
    
    # Tokenize
    logger.info("Tokenizing data...")
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=args.max_length,
            padding='max_length'
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    logger.info("✓ Data tokenized")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",  # Disable wandb, tensorboard
        warmup_steps=100,
        weight_decay=0.01,
        # Memory optimization settings
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="adamw_torch",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    trainer.train()
    
    logger.info("\n✓ Training complete!")
    
    # Save model
    logger.info(f"\nSaving model to {args.model_dir}...")
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    # Save training metrics
    metrics_file = os.path.join(args.output_data_dir, 'training_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    logger.info(f"✓ Model saved to {args.model_dir}")
    logger.info(f"✓ Metrics saved to {metrics_file}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)

if __name__ == '__main__':
    main()