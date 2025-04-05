import torch
from classifier import SiameseNetwork
import os
from tqdm import tqdm
import numpy as np 
import json 

siamese_model = SiameseNetwork(input_dim=1536, hidden_dims=[128]) #input dim=768 for not patentSBERTa
siamese_model.load_state_dict(torch.load("patent_siamese_network_TAC.pt"))
siamese_model.eval()


QUERY_SET = "test"              # Choose from: "train" or "test"
SAVE_RESULTS = True
TOP_N = 100  # Number of documents to retrieve for each query

MODEL_NAME = "PatentSBERTa"  # Choose from: "all-MiniLM-L6-v2" or "PatentSBERTa"
CONTENT_TYPE = "TAC"              # Choose from: "TA", "claims", or "TAC"
POOLING = "mean"                 # The pooling strategy used in create_embeddings.py

BASE_DIR = "/Users/kshitij/Documents/UPSaclay/T4/InfoRetrieval/CodaBench/IR2025"
DOC_EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings/embeddings_precalculated_docs")
TRAIN_EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings/embeddings_precalculated_train")
TEST_EMBEDDING_DIR = os.path.join(BASE_DIR, "embeddings/embeddings_precalculated_test")
OUTPUT_DIR = "/Users/kshitij/Documents/UPSaclay/T4/InfoRetrieval/github/"
CITATION_FILE = os.path.join(BASE_DIR, "Citation_JSONs/Citation_Train.json")

# Embedding files
DOC_EMBEDDING_FILE = os.path.join(DOC_EMBEDDING_DIR, f"embeddings_{MODEL_NAME}_{POOLING}_{CONTENT_TYPE}.npy")
DOC_APP_IDS_FILE = os.path.join(DOC_EMBEDDING_DIR, f"app_ids_{MODEL_NAME}_{POOLING}_{CONTENT_TYPE}.json")

# Select query embedding directory based on QUERY_SET
QUERY_EMBEDDING_DIR = TRAIN_EMBEDDING_DIR if QUERY_SET == "train" else TEST_EMBEDDING_DIR
QUERY_EMBEDDING_FILE = os.path.join(QUERY_EMBEDDING_DIR, f"embeddings_{MODEL_NAME}_{POOLING}_{CONTENT_TYPE}.npy")
QUERY_APP_IDS_FILE = os.path.join(QUERY_EMBEDDING_DIR, f"app_ids_{MODEL_NAME}_{POOLING}_{CONTENT_TYPE}.json")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# === CELL: Function to load embeddings ===
def load_embeddings_and_ids(embedding_file, app_ids_file):
    """
    Load the embeddings and application IDs from saved files
    """
    print(f"Loading embeddings from {embedding_file}")
    embeddings = torch.from_numpy(np.load(embedding_file))

    print(f"Loading app_ids from {app_ids_file}")
    with open(app_ids_file, 'r') as f:
        app_ids = json.load(f)

    print(f"Loaded {len(embeddings)} embeddings and {len(app_ids)} app_ids")
    return embeddings, app_ids

# Load document embeddings and app_ids
doc_embeddings, doc_app_ids = load_embeddings_and_ids(DOC_EMBEDDING_FILE, DOC_APP_IDS_FILE)

# Load query embeddings and app_ids
query_embeddings, query_app_ids = load_embeddings_and_ids(QUERY_EMBEDDING_FILE, QUERY_APP_IDS_FILE)

# Move tensors to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
doc_embeddings = doc_embeddings.to(device)
query_embeddings = query_embeddings.to(device)

print(f"Running retrieval with {len(query_embeddings)} queries against {len(doc_embeddings)} documents")


results = {}
for i, (query_embedding, query_id) in enumerate(tqdm(zip(query_embeddings, query_app_ids), total=len(query_embeddings))):
    # Compute cosine similarity

    query_expanded = query_embedding.unsqueeze(0).repeat(doc_embeddings.shape[0], 1)
    combined = torch.cat((query_expanded, doc_embeddings), dim=1)
    cos_scores = siamese_model.forward(combined).squeeze()
    
    # print(cos_scores.shape)

    # Sort results and get top N
    top_n_index = torch.argsort(cos_scores, descending=True)[:TOP_N].numpy()

    # Get application IDs of top N documents
    top_n_app_ids = [doc_app_ids[i] for i in top_n_index]
    results[query_id] = top_n_app_ids

if QUERY_SET == "train":
    output_file = f"{OUTPUT_DIR}/{MODEL_NAME}_{CONTENT_TYPE}_{QUERY_SET}_retrieved.json"
else:
    output_file = f"{OUTPUT_DIR}/prediction1.json"  # Standard filename for test predictions

if SAVE_RESULTS:
    with open(output_file, 'w') as f:
        json.dump(results, f)
    print(f"Saved retrieval results to {output_file}")
else:
    print(f"Results not saved (SAVE_RESULTS={SAVE_RESULTS})")