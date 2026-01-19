#!/usr/bin/env python3
"""SageMaker training script for fine-tuning Mistral-7B on Igbo language data"""

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
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    return parser.parse_args()

def format_instruction(example):
    if example.get('input'):
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {'text': prompt}

def main():
    args = parse_args()
    logger.info("STARTING IGBO LANGUAGE MODEL TRAINING")
    logger.info(f"Model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
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
    
    logger.info(f"Train path: {args.train}")
    logger.info(f"Files: {os.listdir(args.train)}")
    
    train_dataset = load_dataset('json', data_files=f'{args.train}/*.jsonl', split='train')
    val_dataset = load_dataset('json', data_files=f'{args.validation}/*.jsonl', split='train')
    
    train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_instruction, remove_columns=val_dataset.column_names)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=args.max_length, padding='max_length')
    
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
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
        report_to="none",
        warmup_steps=100,
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        optim="adamw_torch",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    logger.info("STARTING TRAINING")
    trainer.train()
    
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    with open(os.path.join(args.output_data_dir, 'training_metrics.json'), 'w') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    logger.info("TRAINING COMPLETE!")

if __name__ == '__main__':
    main()
