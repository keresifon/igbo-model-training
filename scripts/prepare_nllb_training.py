#!/usr/bin/env python3
"""
Prepare NLLB English-Igbo corpus for LLM fine-tuning
Processes 6.1M sentence pairs from Meta's NLLB dataset
"""

import json
import os
import random
from typing import List, Dict
import time

def load_parallel_corpus(igbo_file: str, english_file: str, 
                         max_pairs: int = None) -> List[Dict]:
    """
    Load NLLB parallel corpus files
    """
    print(f"Loading English from: {english_file}")
    print(f"Loading Igbo from: {igbo_file}")
    print("\nThis may take a few minutes for 6M+ lines...")
    
    start_time = time.time()
    
    # Read files with progress
    print("\nReading English file...")
    with open(english_file, 'r', encoding='utf-8') as f:
        english_lines = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(english_lines):,} English lines")
    
    print("\nReading Igbo file...")
    with open(igbo_file, 'r', encoding='utf-8') as f:
        igbo_lines = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(igbo_lines):,} Igbo lines")
    
    # Verify alignment
    if len(igbo_lines) != len(english_lines):
        print(f"\n⚠️  WARNING: Line count mismatch!")
        print(f"   English: {len(english_lines):,} lines")
        print(f"   Igbo: {len(igbo_lines):,} lines")
        print(f"   Using minimum: {min(len(igbo_lines), len(english_lines)):,}")
    
    # Pair them up
    pairs = []
    num_pairs = min(len(igbo_lines), len(english_lines))
    
    # Apply max_pairs limit if specified
    if max_pairs and max_pairs < num_pairs:
        print(f"\n⚠️  Limiting to {max_pairs:,} pairs (for testing)")
        num_pairs = max_pairs
    
    print(f"\nCreating {num_pairs:,} sentence pairs...")
    for i in range(num_pairs):
        igbo_text = igbo_lines[i]
        english_text = english_lines[i]
        
        # Skip very short pairs
        if len(igbo_text) < 3 or len(english_text) < 3:
            continue
        
        pairs.append({
            'igbo': igbo_text,
            'english': english_text
        })
        
        # Progress indicator
        if (i + 1) % 500000 == 0:
            print(f"  {i + 1:,} / {num_pairs:,} pairs processed...")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Loaded {len(pairs):,} sentence pairs in {elapsed:.1f} seconds")
    
    return pairs

def create_training_examples(data: List[Dict]) -> List[Dict]:
    """
    Convert each Igbo-English pair into 4 instruction-following examples
    """
    print("\nCreating training examples...")
    print(f"This will create {len(data) * 4:,} examples from {len(data):,} pairs")
    
    training_examples = []
    start_time = time.time()
    
    for i, item in enumerate(data):
        igbo_text = item['igbo'].strip()
        english_text = item['english'].strip()
        
        # Create 4 training formats from each pair
        batch = [
            {
                'instruction': 'Translate this Igbo sentence to English:',
                'input': igbo_text,
                'output': english_text
            },
            {
                'instruction': 'Translate this English sentence to Igbo:',
                'input': english_text,
                'output': igbo_text
            },
            {
                'instruction': f'What does "{igbo_text}" mean in English?',
                'input': '',
                'output': f'It means: {english_text}'
            },
            {
                'instruction': f'How do you say "{english_text}" in Igbo?',
                'input': '',
                'output': f'You say: {igbo_text}'
            }
        ]
        
        training_examples.extend(batch)
        
        # Progress indicator
        if (i + 1) % 500000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(data) - i - 1) / rate
            print(f"  {i + 1:,} / {len(data):,} pairs processed... "
                  f"(ETA: {remaining/60:.1f} minutes)")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Created {len(training_examples):,} training examples in {elapsed:.1f} seconds")
    
    return training_examples

def split_train_val_test_grouped(training_examples: List[Dict], 
                                 examples_per_pair: int = 4) -> tuple:
    """
    Split keeping the 4 examples from each sentence pair together
    """
    print("\nSplitting dataset...")
    
    # Group examples (every 4 consecutive = 1 sentence pair)
    groups = []
    for i in range(0, len(training_examples), examples_per_pair):
        group = training_examples[i:i+examples_per_pair]
        if len(group) == examples_per_pair:
            groups.append(group)
    
    print(f"Grouped {len(training_examples):,} examples into {len(groups):,} groups")
    
    # Shuffle groups
    print("Shuffling groups...")
    random.shuffle(groups)
    
    # Split groups
    total_groups = len(groups)
    train_end = int(total_groups * 0.8)
    val_end = train_end + int(total_groups * 0.1)
    
    train_groups = groups[:train_end]
    val_groups = groups[train_end:val_end]
    test_groups = groups[val_end:]
    
    # Flatten back to individual examples
    print("Flattening groups...")
    train_data = [ex for group in train_groups for ex in group]
    val_data = [ex for group in val_groups for ex in group]
    test_data = [ex for group in test_groups for ex in group]
    
    total = len(train_data) + len(val_data) + len(test_data)
    
    print(f"\nDataset split:")
    print(f"  Training:   {len(train_data):,} examples ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data):,} examples ({len(val_data)/total*100:.1f}%)")
    print(f"  Test:       {len(test_data):,} examples ({len(test_data)/total*100:.1f}%)")
    
    return train_data, val_data, test_data

def save_jsonl(data: List[Dict], filepath: str):
    """Save data in JSONL format with progress"""
    print(f"\nSaving {len(data):,} examples to {filepath}...")
    
    start_time = time.time()
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            if (i + 1) % 1000000 == 0:
                print(f"  Written {i + 1:,} / {len(data):,} examples...")
    
    elapsed = time.time() - start_time
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"✓ Saved in {elapsed:.1f}s - {size_mb:.1f} MB")

def main():
    # Configuration
    IGBO_FILE = './en-ig/NLLB.en-ig.ig'
    ENGLISH_FILE = './en-ig/NLLB.en-ig.en'
    OUTPUT_DIR = './datasets/processed-nllb'
    
    # Optional: Set max_pairs for testing (set to None for full dataset)
    # MAX_PAIRS = 100000  # Uncomment to test with 100K pairs first
    MAX_PAIRS = None  # Use full dataset
    
    # Check files exist
    if not os.path.exists(IGBO_FILE):
        print(f"❌ Igbo file not found: {IGBO_FILE}")
        print("\nPlease run download_nllb.sh first:")
        print("  chmod +x download_nllb.sh")
        print("  ./download_nllb.sh")
        return
    
    if not os.path.exists(ENGLISH_FILE):
        print(f"❌ English file not found: {ENGLISH_FILE}")
        return
    
    # Set random seed
    random.seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("NLLB ENGLISH-IGBO DATASET PREPARATION")
    print("="*70)
    print("\nThis will process ~6.1 million sentence pairs")
    print("Estimated time: 30-60 minutes")
    print("Estimated output size: 10-15 GB")
    print("="*70)
    
    # Step 1: Load data
    raw_data = load_parallel_corpus(IGBO_FILE, ENGLISH_FILE, MAX_PAIRS)
    
    # Show samples
    print("\n" + "="*70)
    print("SAMPLE SENTENCE PAIRS")
    print("="*70)
    for i in range(min(3, len(raw_data))):
        print(f"\n{i+1}. English: {raw_data[i]['english']}")
        print(f"   Igbo:    {raw_data[i]['igbo']}")
    
    # Step 2: Create training examples
    print("\n" + "="*70)
    print("CREATING TRAINING EXAMPLES")
    print("="*70)
    training_data = create_training_examples(raw_data)
    
    # Step 3: Split dataset
    print("\n" + "="*70)
    print("SPLITTING DATASET")
    print("="*70)
    train_data, val_data, test_data = split_train_val_test_grouped(training_data)
    
    # Step 4: Save files
    print("\n" + "="*70)
    print("SAVING FILES")
    print("="*70)
    
    save_jsonl(train_data, f'{OUTPUT_DIR}/nllb_train.jsonl')
    save_jsonl(val_data, f'{OUTPUT_DIR}/nllb_val.jsonl')
    save_jsonl(test_data, f'{OUTPUT_DIR}/nllb_test.jsonl')
    
    # Save statistics
    stats = {
        'source': 'NLLB v1 (OPUS)',
        'total_sentence_pairs': len(raw_data),
        'total_training_examples': len(training_data),
        'splits': {
            'train': len(train_data),
            'validation': len(val_data),
            'test': len(test_data)
        },
        'sample_pairs': raw_data[:5],
        'sample_training': train_data[:10]
    }
    
    with open(f'{OUTPUT_DIR}/nllb_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved statistics to {OUTPUT_DIR}/nllb_stats.json")
    
    # Final summary
    print("\n" + "="*70)
    print("PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nDataset: NLLB English-Igbo")
    print(f"Sentence pairs: {len(raw_data):,}")
    print(f"Training examples: {len(training_data):,}")
    print(f"\nFiles created in {OUTPUT_DIR}/:")
    print(f"  • nllb_train.jsonl ({len(train_data):,} examples)")
    print(f"  • nllb_val.jsonl ({len(val_data):,} examples)")
    print(f"  • nllb_test.jsonl ({len(test_data):,} examples)")
    print(f"  • nllb_stats.json (statistics)")
    
    # Calculate sizes
    total_size = sum(
        os.path.getsize(f'{OUTPUT_DIR}/{f}') 
        for f in ['nllb_train.jsonl', 'nllb_val.jsonl', 'nllb_test.jsonl']
    ) / (1024 ** 3)
    
    print(f"\nTotal size: {total_size:.2f} GB")
    
    # Estimate costs
    print("\n" + "="*70)
    print("ESTIMATED TRAINING COSTS")
    print("="*70)
    hours_estimate = 150
    cost_regular = hours_estimate * 1.41
    cost_spot = hours_estimate * 0.42
    
    print(f"\nTraining examples: {len(training_data):,}")
    print(f"Estimated training time: ~{hours_estimate} hours (6 days)")
    print(f"\nInstance: ml.g5.xlarge")
    print(f"  Regular pricing: ${cost_regular:.2f}")
    print(f"  Spot pricing (70% off): ${cost_spot:.2f}")
    print(f"\nS3 storage: ~$1/month")
    print(f"\nTotal first-time cost: ${cost_spot:.2f} - ${cost_spot + 10:.2f}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\n1. Upload to S3:")
    print(f"   aws s3 sync {OUTPUT_DIR} s3://kere-igbo-learning-project/datasets/nllb/")
    print("\n2. Update launch_sagemaker_training.py:")
    print("   Change train_data_s3 to point to s3://.../datasets/nllb/nllb_train.jsonl")
    print("\n3. Launch training:")
    print("   python3 launch_sagemaker_training.py")
    print("\nEnjoy your world-class Igbo language model! 🎉")

if __name__ == '__main__':
    main()
