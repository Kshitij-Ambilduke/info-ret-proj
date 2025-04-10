import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def load_json_file(file_path):
    """Load JSON data from a file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """Save data to a JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_content_data(file_path):
    """Load content data from a JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create a dictionary mapping FAN to Content
    content_dict = {item['FAN']: item['Content'] for item in data}
    return content_dict

def extract_text(content_dict, text_type="full"):
    """Extract text from patent content based on text_type"""
    if text_type == "TA" or text_type == "title_abstract":
        # Extract title and abstract
        title = content_dict.get("title", "")
        abstract = content_dict.get("pa01", "")
        return f"{title} {abstract}".strip()
    
    elif text_type == "claims":
        # Extract all claims (keys starting with 'c')
        claims = []
        for key, value in content_dict.items():
            if key.startswith('c-'):
                claims.append(value)
        return " ".join(claims)

    elif text_type == "tac1":
        # Extract title, abstract, and first claim
        title = content_dict.get("title", "")
        abstract = content_dict.get("pa01", "")
        # Find the first claim safely
        first_claim = ""
        for key, value in content_dict.items():
            if key.startswith('c-'):
                first_claim = value
                break
        return f"{title} {abstract} {first_claim}".strip()
    
    elif text_type == "description":
        # Extract all paragraphs (keys starting with 'p')
        paragraphs = []
        for key, value in content_dict.items():
            if key.startswith('p'):
                paragraphs.append(value)
        return " ".join(paragraphs)
    
    elif text_type == "full":
        # Extract everything
        all_text = []
        # Start with title and abstract for better context at the beginning
        if "title" in content_dict:
            all_text.append(content_dict["title"])
        if "pa01" in content_dict:
            all_text.append(content_dict["pa01"])
            
        # Add claims and description
        for key, value in content_dict.items():
            if key != "title" and key != "pa01":
                all_text.append(value)
                
        return " ".join(all_text)
    
    return ""

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract the last token representations for pooling"""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create an instruction-formatted query"""
    return f'Instruct: {task_description}\nQuery: {query}'

def cross_encoder_reranking(query_text, doc_texts, model, tokenizer, batch_size=8, max_length=2048):
    """
    Rerank document texts based on query text using cross-encoder model
    
    Parameters:
    query_text (str): The query text
    doc_texts (list): List of document texts
    model: The cross-encoder model
    tokenizer: The tokenizer for the model
    batch_size (int): Batch size for processing
    max_length (int): Maximum sequence length
    
    Returns:
    list: Indices of documents sorted by relevance score (descending)
    """
    device = next(model.parameters()).device
    scores = []
    
    # Format query with instruction
    task_description = 'Re-rank a set of retrieved patents based on their relevance to a given query patent. The task aims to refine the order of patents by evaluating their semantic similarity to the query patent, ensuring that the most relevant patents appear at the top of the list.'

    instructed_query = get_detailed_instruct(task_description, query_text)
    
    # Process in batches to avoid OOM
    for i in tqdm(range(0, len(doc_texts), batch_size), desc="Scoring documents", leave=False):
        batch_docs = doc_texts[i:i+batch_size]
        
        # Prepare input pairs for the batch
        input_texts = [instructed_query] + batch_docs
        
        # Tokenize
        with torch.no_grad():
            batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, 
                                  truncation=True, return_tensors='pt').to(device)
            
            # Get embeddings
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Calculate similarity scores between query and documents
            batch_scores = (embeddings[0].unsqueeze(0) @ embeddings[1:].T).squeeze(0) * 100
            scores.extend(batch_scores.cpu().tolist())
    
    # Create list of (index, score) tuples for sorting
    indexed_scores = list(enumerate(scores))
    
    # Sort by score in descending order
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return sorted indices
    return indexed_scores

def main():
    parser = argparse.ArgumentParser(description='Re-rank patents using cross-encoder scoring (training queries only)')
    parser.add_argument('--pre_ranking', type=str, default='shuffled_pre_ranking.json',
                        help='Path to pre-ranking JSON file')
    parser.add_argument('--output', type=str, default='predictions2.json',
                        help='Path to output re-ranked JSON file')
    parser.add_argument('--queries_content', type=str, 
                        default='./queries_content_with_features.json',
                        help='Path to queries content JSON file')
    parser.add_argument('--documents_content', type=str, 
                        default='./documents_content_with_features.json',
                        help='Path to documents content JSON file')
    parser.add_argument('--queries_list', type=str, default='train_queries.json',
                        help='Path to training queries JSON file')
    parser.add_argument('--text_type', type=str, default='claims',
                        choices=['TA', 'claims', 'description', 'full', 'tac1'],
                        help='Type of text to use for scoring')
    parser.add_argument('--model_name', type=str, default="anferico/bert-for-patents", 
                        help='Name of the cross-encoder model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for scoring')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--base_dir', type=str, 
                        default='/content/out',
                        help='Base directory for data files')
    
    args = parser.parse_args()
    
    # Ensure all paths are relative to base_dir if they're not absolute
    def get_full_path(path):
        if os.path.isabs(path):
            return path
        return os.path.join(args.base_dir, path)
    
    # Load training queries
    print(f"Loading training queries from {args.queries_list}...")
    queries_list = load_json_file(get_full_path(args.queries_list))
    print(f"Loaded {len(queries_list)} training queries")
    
    # Load pre-ranking data
    print(f"Loading pre-ranking data from {args.pre_ranking}...")
    pre_ranking = load_json_file(get_full_path(args.pre_ranking))
    
    # Filter pre-ranking to include only training queries
    pre_ranking = {fan: docs for fan, docs in pre_ranking.items() if fan in queries_list}
    print(f"Filtered pre-ranking to {len(pre_ranking)} training queries")
    
    # Load content data
    print(f"Loading query content from {args.queries_content}...")
    queries_content = load_content_data(get_full_path(args.queries_content))
    
    print(f"Loading document content from {args.documents_content}...")
    documents_content = load_content_data(get_full_path(args.documents_content))
    
    # Load model and tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()
    
    # Process each query and re-rank its documents
    print("Starting re-ranking process for training queries...")
    re_ranked = {}
    missing_query_fans = []
    missing_doc_fans = {}
    
    
    for query_fan, pre_ranked_docs in tqdm(pre_ranking.items(), desc="Processing queries"):
        # Check if query FAN exists in our content data
        if query_fan not in queries_content:
            missing_query_fans.append(query_fan)
            continue
        
        # Extract query text
        query_text = extract_text(queries_content[query_fan], args.text_type)
        if not query_text:
            missing_query_fans.append(query_fan)
            continue

        
            
        # Prepare document texts and keep track of their fans
        doc_texts = []
        doc_fans = []
        missing_docs_for_query = []
        
        for doc_fan in pre_ranked_docs:
            if doc_fan not in documents_content:
                missing_docs_for_query.append(doc_fan)
                continue
                
            doc_text = extract_text(documents_content[doc_fan], args.text_type)
            if doc_text:
                doc_texts.append(doc_text)
                doc_fans.append(doc_fan)
        
        # Keep track of missing documents
        if missing_docs_for_query:
            missing_doc_fans[query_fan] = missing_docs_for_query
            
        # Skip if no valid documents
        if not doc_texts:
            re_ranked[query_fan] = []
            continue
            
        # Re-rank documents
        print(f"\nRe-ranking {len(doc_texts)} documents for training query {query_fan}")
        
        # Print some of the original pre-ranking order for debugging
        print(f"Original pre-ranking (first 3): {doc_fans[:3]}")
        
        ##################################################################
            # Re-rank documents and get scores
        sorted_scores = cross_encoder_reranking(
            query_text, doc_texts, model, tokenizer, 
            batch_size=args.batch_size, max_length=args.max_length
        )
        
        # Create list of {patent_id, score} dictionaries in sorted order
        re_ranked_docs = []
        for idx, score in sorted_scores:
            re_ranked_docs.append({
                "patent_id": doc_fans[idx],
                "score": score
            })
        re_ranked[query_fan] = re_ranked_docs
        ##################################################################

    # Report any missing FANs
    if missing_query_fans:
        print(f"Warning: {len(missing_query_fans)} query FANs were not found in the content data")
    if missing_doc_fans:
        total_missing = sum(len(docs) for docs in missing_doc_fans.values())
        print(f"Warning: {total_missing} document FANs were not found in the content data")
    
    # Save re-ranked results
    output_path = get_full_path(args.output)
    print(f"Saving re-ranked results to {output_path}...")
    save_json_file(re_ranked, output_path)
    
    print("Re-ranking complete!")
    print(f"Number of training queries processed: {len(re_ranked)}")
    
    # Optionally save the missing FANs information for debugging
    if missing_query_fans or missing_doc_fans:
        missing_info = {
            "missing_query_fans": missing_query_fans,
            "missing_doc_fans": missing_doc_fans
        }
        missing_info_path = f"{os.path.splitext(output_path)[0]}_missing_fans.json"
        save_json_file(missing_info, missing_info_path)
        print(f"Information about missing FANs saved to {missing_info_path}")


if __name__=="__main__":
    print("ljbdkabdkabkjdbasuobdondpad")
    main()