import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloading import CustomDataset
from sklearn.metrics import average_precision_score
from tqdm import tqdm

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        """
        A Siamese network for patent similarity classification.
        
        Args:
            input_dim: Dimension of the concatenated embedding vectors
            hidden_dims: List of hidden layer dimensions
        """
        super(SiameseNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.model(x.squeeze())


def train_siamese_network(train_dataset, val_loader, model, num_epochs=20, lr=0.001, weight_decay=1e-7):
    """
    Train the Siamese network
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: SiameseNetwork instance
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
    
    Returns:
        Trained model and training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    print("model created")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ap': []
    }
    
    best_val_ap = 0.0
    best_model = None
    
    for epoch in tqdm(range(num_epochs)):
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) ## every epoch new negatives will be considered
    
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (embeddings, labels) in enumerate(train_loader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(embeddings).squeeze()
            loss = criterion(outputs, labels.float())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_ap = average_precision_score(all_labels, all_preds)
        
        history['val_loss'].append(avg_val_loss)
        history['val_ap'].append(val_ap)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val AP: {val_ap:.4f}")
        
        # Save best model
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_model = model.state_dict().copy()
            print(f"New best model saved with AP: {val_ap:.4f}")
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, history

if __name__ == "__main__":
    
    # Paths
    base_dir = "/Users/kshitij/Documents/UPSaclay/T4/InfoRetrieval/CodaBench/IR2025"
    model_name = "all-MiniLM-L6-v2"
    data_type = "TA"
    
    # Create datasets
    train_dataset = CustomDataset(model=model_name, which_data_incoming=data_type, 
                                which_data_existing=data_type, base_dir=base_dir, split="train")
    
    # Create a validation set from training data (80-20 split)
    train_size = int(0.5 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    _, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=0)
    
    # Get embedding dimension from the first batch
    for embeddings, _ in val_loader:
        input_dim = embeddings.shape[-1]
        break
    
    # Create and train model
    siamese_model = SiameseNetwork(input_dim=input_dim, hidden_dims=[512, 256, 128])
    trained_model, history = train_siamese_network(train_dataset, val_loader, siamese_model, 
                                                  num_epochs=50, lr=0.001)
        
    print(history)
    
    # Save the model
    torch.save(trained_model.state_dict(), "patent_siamese_network.pt")