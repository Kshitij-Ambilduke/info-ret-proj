import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load PatentsBERTa model and tokenizer
model_name = "AI-Growth-Lab/PatentSBERTa"  # Verify correct model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_embeddings(texts, model, tokenizer, batch_size=32):
    """Convert texts to sentence embeddings using mean pooling"""
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs.attention_mask
            
            # Mean pooling with attention masking
            masked_hidden = last_hidden_state * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(-1)
            
            embeddings.append(pooled.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0)

def extract_sentences(patent_data):
    """Extract sentences with metadata from patent data"""
    sentences = []
    for item in patent_data:
        fan = item["FAN"]
        for content_key, text in item["Content"].items():
            if content_key == "title":
                continue
            sentences.append({
                "fan": fan,
                "content_key": content_key,
                "text": text
            })
    return sentences

# Load your patent data (replace with actual data loading)
# existing_patent_data = [...]  # Load existing patent data
with open("/Users/kshitij/Documents/UPSaclay/T4/InfoRetrieval/CodaBench/IR2025/Task2\ starter\ pack/queries_content_with_features.json") as f:
    incoming_patent_data = json.load(f)
with open("/Users/kshitij/Documents/UPSaclay/T4/InfoRetrieval/CodaBench/IR2025/Task2\ starter\ pack/documents_content_with_features.json") as f:
    existing_patent_data = json.load(f)

# existing_patent_data = 

# incoming_patent_data = [...]  # Load incoming patent data

# Extract sentences with metadata
existing_sentences = extract_sentences(existing_patent_data)
incoming_sentences = extract_sentences(incoming_patent_data)

# Generate embeddings
existing_embeddings = get_embeddings(
    [s["text"] for s in existing_sentences], model, tokenizer
)
incoming_embeddings = get_embeddings(
    [s["text"] for s in incoming_sentences], model, tokenizer
)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(existing_embeddings, incoming_embeddings)

# Get top K pairs
K = 10  # Set your desired K value
indices = np.argsort(similarity_matrix.ravel())[-K:][::-1]
rows, cols = np.unravel_index(indices, similarity_matrix.shape)

# Prepare results
top_pairs = []
for row, col in zip(rows, cols):
    existing = existing_sentences[row]
    incoming = incoming_sentences[col]
    top_pairs.append({
        "existing": {
            "FAN": existing["fan"],
            "content_key": existing["content_key"],
            "text": existing["text"]
        },
        "incoming": {
            "FAN": incoming["fan"],
            "content_key": incoming["content_key"],
            "text": incoming["text"]
        },
        "similarity_score": similarity_matrix[row, col]
    })

# Sort by descending similarity score
top_pairs.sort(key=lambda x: x["similarity_score"], reverse=True)

# Print results
for i, pair in enumerate(top_pairs, 1):
    print(f"Pair {i}: Similarity = {pair['similarity_score']:.4f}")
    print(f"Existing ({pair['existing']['FAN']}/{pair['existing']['content_key']}):")
    print(pair['existing']['text'][:100] + "...")
    print(f"Incoming ({pair['incoming']['FAN']}/{pair['incoming']['content_key']}):")
    print(pair['incoming']['text'][:100] + "...")
    print("\n" + "-"*80 + "\n")