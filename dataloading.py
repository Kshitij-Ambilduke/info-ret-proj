import torch
from torch.utils.data import Dataset, DataLoader
import os
import json 
import numpy as np
import random

class CustomDataset(Dataset):
    def __init__(self, model, which_data_incoming, which_data_existing, base_dir, split='train'):
        '''
        args:
            split = train/test
            base_dir = /Users/kshitij/Documents/UPSaclay/T4/InfoRetrieval/CodaBench/IR2025
            which_data_incoming = "TA"/"claims"/"TAC"
            which_data_existing = "TA"/"claims"/"TAC"
            model = "all-MiniLM-L6-v2" or "PatentSBERTa"
        '''

        doc_embedding_dir = os.path.join(base_dir, "embeddings/embeddings_precalculated_docs")
        split_embedding_dir =  os.path.join(base_dir, f"embeddings/embeddings_precalculated_{split}")
        
        self.existing_patent_embeddings = os.path.join(doc_embedding_dir, f"embeddings_{model}_mean_{which_data_existing}.npy")
        self.existing_patent_ids = os.path.join(doc_embedding_dir, f"app_ids_{model}_mean_{which_data_existing}.json")

        self.incoming_patent_embeddings = os.path.join(split_embedding_dir, f"embeddings_{model}_mean_{which_data_incoming}.npy")
        self.incoming_patent_ids = os.path.join(split_embedding_dir, f"app_ids_{model}_mean_{which_data_incoming}.json")

        if split=='train':
            citation_file = os.path.join(base_dir, "Citation_JSONs/Citation_Train.json")
            with open(citation_file, 'r') as f:
                citations = json.load(f)
            mapping_dict = self.citation_to_citing_to_cited_dict(citations)
        
            self.positive_pairs_list = [] # [..,(i ,j),..] i - incoming patent, j - existing patent
            for i in mapping_dict:
                for j in mapping_dict[i]:
                    self.positive_pairs_list.append((i,j))

            self.num_positives = len(self.positive_pairs_list)
            
            self.incoming_embeddings, self.incoming_app_ids = self.load_embeddings_and_ids(self.incoming_patent_embeddings, self.incoming_patent_ids)
            self.existing_embeddings, self.existing_app_ids = self.load_embeddings_and_ids(self.existing_patent_embeddings, self.existing_patent_ids)
            
            # store negative examples for sampling dynamically
            self.negative_candidates = {}
            for incoming_id in mapping_dict:
                documents_not_to_consider = mapping_dict[incoming_id]
                self.negative_candidates[incoming_id] = list(set(self.existing_app_ids) - set(documents_not_to_consider))
        
        # print(self.positive_pairs_list[0].shape)
            

    def __len__(self):
        # Return 2x the number of positives (1 positive + 1 negative per positive pair)
        return 2 * self.num_positives
    
    def __getitem__(self, idx):
        if idx < self.num_positives:
            # Return positive pair
            incoming_id, existing_id = self.positive_pairs_list[idx]
            label = 1
        else:
            # Return negative pair (dynamic sampling)
            pair_idx = idx - self.num_positives
            incoming_id, _ = self.positive_pairs_list[pair_idx]
            candidates = self.negative_candidates[incoming_id]
            existing_id = random.choice(candidates)
            label = 0
        
        incoming_embed = self.get_embedding_incoming(self.incoming_embeddings, self.incoming_app_ids, incoming_id)
        existing_embed = self.get_embedding_existing(self.existing_embeddings, self.existing_app_ids, existing_id)
        concatenated = torch.cat((incoming_embed, existing_embed), dim=1)

        return concatenated, label
    
    def load_embeddings_and_ids(self, embedding_file, app_ids_file):
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

    def get_embedding_incoming(self, incoming_embeddings, incoming_app_ids, incoming_id):
        index_of_incoming_id = incoming_app_ids.index(incoming_id)
        return incoming_embeddings[index_of_incoming_id].unsqueeze(0)

    def get_embedding_existing(self, existing_embeddings, existing_app_ids, existing_id):
        index_of_existing_id = existing_app_ids.index(existing_id)
        return existing_embeddings[index_of_existing_id].unsqueeze(0)

    def citation_to_citing_to_cited_dict(self, citations):
        """
        Put a citation mapping in a dict format
        """
        # Initialize an empty dictionary to store the results
        citing_to_cited_dict = {}

        # Iterate over the items in the JSON list
        for citation in citations:
            # Check if the citing id already exists in the resulting dictionary
            if citation[0] in citing_to_cited_dict:
                # If the citing id exists, append the cited id to the existing list
                citing_to_cited_dict[citation[0]].append(citation[2])
            else:
                # If the citing id doesn't exist, create a new list with the cited id for that citing id
                citing_to_cited_dict[citation[0]] = [citation[2]]

        return citing_to_cited_dict

def collator_func(batch):
    pass
    
# dataset = CustomDataset(
#     model = "all-MiniLM-L6-v2",
#     which_data_incoming = "TA",
#     which_data_existing = "TA",
#     base_dir = "/Users/kshitij/Documents/UPSaclay/T4/InfoRetrieval/CodaBench/IR2025",
#     split = "train"
# )
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers=0 for dynamic sampling
# for i in dataloader:
#     print(i[0].shape, i[1])
#     # print(i[1])
#     break
    