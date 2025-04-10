
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class PatentSimilarityFilter:
    def __init__(self, model_path, threshold=0.7):
        """
        Initialize the patent similarity filter
        
        Args:
            model_path: Path to the fine-tuned PatentsBERTa model
            threshold: Confidence threshold for considering content as similar
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.threshold = threshold
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict_similarity(self, text_a, text_b):
        """
        Predict similarity between two text segments
        
        Returns:
            is_similar: Boolean indicating if texts are similar
            confidence: Confidence score for the prediction
        """
        inputs = self.tokenizer(
            text_a, 
            text_b, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        is_similar = (prediction == 1 and confidence >= self.threshold)
        
        return is_similar, confidence, prediction
    
    def filter_patent_pairs(self, patent_a, patent_b):
        """
        Extract similar content between two patents
        
        Args:
            patent_a: Dictionary with patent content
            patent_b: Dictionary with patent content
            
        Returns:
            similar_pairs: List of tuples containing similar content pairs
        """
        similar_pairs = []
        
        # Extract content IDs
        patent_a_ids = [k for k in patent_a.keys() if k.startswith(('c-', 'p'))]
        patent_b_ids = [k for k in patent_b.keys() if k.startswith(('c-', 'p'))]
        
        # Compare all pairs
        for id_a in patent_a_ids:
            text_a = patent_a[id_a]
            
            for id_b in patent_b_ids:
                text_b = patent_b[id_b]
                
                is_similar, confidence, prediction = self.predict_similarity(text_a, text_b)
                
                if is_similar:
                    similar_pairs.append({
                        'id_a': id_a,
                        'id_b': id_b,
                        'text_a': text_a,
                        'text_b': text_b,
                        'confidence': confidence
                    })
        
        # Sort by confidence
        similar_pairs = sorted(similar_pairs, key=lambda x: x['confidence'], reverse=True)
        
        return similar_pairs

def batch_process_for_reranking(model_path, citing_df, nonciting_df, output_path=None, threshold=0.7, batch_size=100):
    """
    Process patents in batches to find similar content for reranking
    
    Args:
        model_path: Path to fine-tuned model
        citing_df: DataFrame with citing patents
        nonciting_df: DataFrame with non-citing patents
        output_path: Path to save results (optional)
        threshold: Similarity confidence threshold
        batch_size: Number of patent pairs to process at once
    
    Returns:
        results_df: DataFrame with similar content pairs
    """
    similarity_filter = PatentSimilarityFilter(model_path, threshold)
    
    # Get all application numbers
    citing_apps = list(citing_df.index)
    nonciting_apps = list(nonciting_df.index)
    
    results = []
    
    # Process in batches for memory efficiency
    for i in tqdm(range(0, min(len(citing_apps), batch_size))):
        citing_app = citing_apps[i]
        citing_content = citing_df.loc[citing_app]["Content"]
        
        for j in range(0, min(len(nonciting_apps), batch_size)):
            nonciting_app = nonciting_apps[j]
            nonciting_content = nonciting_df.loc[nonciting_app]["Content"]
            
            similar_pairs = similarity_filter.filter_patent_pairs(
                citing_content, nonciting_content
            )
            
            for pair in similar_pairs:
                results.append({
                    'citing_app': citing_app,
                    'nonciting_app': nonciting_app,
                    'citing_id': pair['id_a'],
                    'nonciting_id': pair['id_b'],
                    'citing_text': pair['text_a'],
                    'nonciting_text': pair['text_b'],
                    'confidence': pair['confidence']
                })
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Save results if output_path is provided
    if output_path:
        results_df.to_csv(output_path, index=False)
    
    return results_df

# Example usage
if __name__ == "__main__":
    # Load your data
    citing_dataset_df = pd.read_pickle("data_task1/citing_dataset.pkl")
    nonciting_dataset_df = pd.read_pickle("data_task1/nonciting_dataset.pkl")
    
    # Path to fine-tuned model
    model_path = "./patent_match_model/best_model"
    
    # Process patent pairs
    similar_content_df = batch_process_for_reranking(
        model_path=model_path,
        citing_df=citing_dataset_df,
        nonciting_df=nonciting_dataset_df,
        output_path="similar_content_for_reranking.csv",
        threshold=0.7,
        batch_size=50  # Adjust based on your computational resources
    )
    
    print(f"Found {len(similar_content_df)} similar content pairs for reranking")
    
    # Display sample results
    if len(similar_content_df) > 0:
        print("\nSample results:")
        print(similar_content_df.head())

