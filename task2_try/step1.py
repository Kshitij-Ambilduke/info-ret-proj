import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm

# Load the PatentsBERTa tokenizer and model
model_name = "ai-r/PatentsBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

class PatentMatchDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        encoding = self.tokenizer(
            example["text_a"],
            example["text_b"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(example["label"], dtype=torch.long)
        }

def prepare_training_data(citing_df, nonciting_df, mapping_df):
    """
    Prepare training data from the provided dataframes
    """
    examples = []
    
    # Create positive examples from mapping_df (label=1)
    for _, row in tqdm(mapping_df.iterrows(), desc="Processing positive examples"):
        citing_app_num = row[0]
        citing_claim_ids = row[1]
        nonciting_app_num = row[2]
        nonciting_ids = row[3]
        
        # Skip if any of the required data is missing
        if citing_app_num not in citing_df.index or nonciting_app_num not in nonciting_df.index:
            continue
            
        citing_content = citing_df.loc[citing_app_num]["Content"]
        nonciting_content = nonciting_df.loc[nonciting_app_num]["Content"]
        
        # Create positive pairs
        for citing_id in citing_claim_ids:
            if citing_id not in citing_content:
                continue
                
            citing_text = citing_content[citing_id]
            
            for nonciting_id in nonciting_ids:
                if nonciting_id not in nonciting_content:
                    continue
                    
                nonciting_text = nonciting_content[nonciting_id]
                
                examples.append({
                    "text_a": citing_text,
                    "text_b": nonciting_text,
                    "label": 1,
                    "citing_app": citing_app_num,
                    "nonciting_app": nonciting_app_num
                })
    
    # Create negative examples (label=0)
    # Strategy: For each positive pair, create negative pairs from the same patents but with non-matching content
    positive_pairs = set([(ex["citing_app"], ex["nonciting_app"]) for ex in examples])
    negative_examples = []
    
    # 1. Similar patent negative examples
    for citing_app_num, nonciting_app_num in tqdm(positive_pairs, desc="Processing negative examples"):
        citing_content = citing_df.loc[citing_app_num]["Content"]
        nonciting_content = nonciting_df.loc[nonciting_app_num]["Content"]
        
        # Get all citation IDs and non-citation IDs
        citing_ids = [k for k in citing_content.keys() if k.startswith(('c-', 'p'))]
        nonciting_ids = [k for k in nonciting_content.keys() if k.startswith(('c-', 'p'))]
        
        # Find matching pairs from the positive examples
        matching_pairs = []
        for ex in examples:
            if ex["citing_app"] == citing_app_num and ex["nonciting_app"] == nonciting_app_num:
                for citing_id in citing_ids:
                    if citing_content[citing_id] == ex["text_a"]:
                        for nonciting_id in nonciting_ids:
                            if nonciting_content[nonciting_id] == ex["text_b"]:
                                matching_pairs.append((citing_id, nonciting_id))
        
        # Create negative examples by sampling non-matching content
        sampled_citing_ids = random.sample(citing_ids, min(len(citing_ids), 5))
        sampled_nonciting_ids = random.sample(nonciting_ids, min(len(nonciting_ids), 5))
        
        for citing_id in sampled_citing_ids:
            for nonciting_id in sampled_nonciting_ids:
                if (citing_id, nonciting_id) not in matching_pairs:
                    negative_examples.append({
                        "text_a": citing_content[citing_id],
                        "text_b": nonciting_content[nonciting_id],
                        "label": 0,
                        "citing_app": citing_app_num,
                        "nonciting_app": nonciting_app_num
                    })
    
    # 2. Dissimilar patent negative examples
    # Sample from patents that don't have any mappings between them
    all_citing_apps = set(citing_df.index)
    all_nonciting_apps = set(nonciting_df.index)
    
    # Get patents that are not in positive pairs
    unpaired_citing_apps = random.sample(list(all_citing_apps - set([p[0] for p in positive_pairs])), 
                                         min(len(all_citing_apps - set([p[0] for p in positive_pairs])), 20))
    unpaired_nonciting_apps = random.sample(list(all_nonciting_apps - set([p[1] for p in positive_pairs])), 
                                            min(len(all_nonciting_apps - set([p[1] for p in positive_pairs])), 20))
    
    # Create dissimilar patent negative examples
    for citing_app_num in tqdm(unpaired_citing_apps, desc="Processing dissimilar patent examples"):
        for nonciting_app_num in unpaired_nonciting_apps:
            if citing_app_num in citing_df.index and nonciting_app_num in nonciting_df.index:
                citing_content = citing_df.loc[citing_app_num]["Content"]
                nonciting_content = nonciting_df.loc[nonciting_app_num]["Content"]
                
                citing_ids = [k for k in citing_content.keys() if k.startswith(('c-', 'p'))]
                nonciting_ids = [k for k in nonciting_content.keys() if k.startswith(('c-', 'p'))]
                
                # Sample a few elements to create negative pairs
                if citing_ids and nonciting_ids:
                    citing_id = random.choice(citing_ids)
                    nonciting_id = random.choice(nonciting_ids)
                    
                    negative_examples.append({
                        "text_a": citing_content[citing_id],
                        "text_b": nonciting_content[nonciting_id],
                        "label": 0,
                        "citing_app": citing_app_num,
                        "nonciting_app": nonciting_app_num
                    })
    
    # Combine positive and negative examples
    all_examples = examples + negative_examples
    
    # Balance the dataset (optional but recommended)
    num_positives = len(examples)
    num_negatives = len(negative_examples)
    
    if num_negatives > num_positives * 3:  # Limit negative examples to 3x positive examples
        negative_examples = random.sample(negative_examples, num_positives * 3)
        all_examples = examples + negative_examples
    
    print(f"Dataset stats: {len(examples)} positive examples, {len(negative_examples)} negative examples")
    
    return all_examples

def train_model(train_dataset, val_dataset, output_dir, batch_size=8, num_epochs=3, learning_rate=2e-5):
    """
    Train the PatentsBERTa model
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    trainer.save_model(f"{output_dir}/best_model")
    tokenizer.save_pretrained(f"{output_dir}/best_model")
    
    return trainer

def main():
    # Load datasets
    print("Loading datasets...")
    citing_dataset_df = pd.read_pickle("citing_dataset.pkl")
    nonciting_dataset_df = pd.read_pickle("nonciting_dataset.pkl")
    mapping_dataset_df = pd.read_pickle("mapping_dataset.pkl")
    
    # Prepare training data
    print("Preparing training data...")
    all_examples = prepare_training_data(citing_dataset_df, nonciting_dataset_df, mapping_dataset_df)
    
    # Split data into train and validation sets
    train_examples, val_examples = train_test_split(all_examples, test_size=0.1, random_state=42)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = PatentMatchDataset(train_examples, tokenizer)
    val_dataset = PatentMatchDataset(val_examples, tokenizer)
    
    # Train model
    print("Training model...")
    output_dir = "./patent_match_model"
    trainer = train_model(train_dataset, val_dataset, output_dir)
    
    print("Training complete!")
    
    # Example of how to use the model for inference
    def predict_similarity(text_a, text_b):
        inputs = tokenizer(text_a, text_b, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        return prediction, confidence
    
    # Example usage
    print("\nExample prediction:")
    sample = val_examples[0]
    prediction, confidence = predict_similarity(sample["text_a"], sample["text_b"])
    print(f"Prediction: {prediction} (Ground truth: {sample['label']})")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()