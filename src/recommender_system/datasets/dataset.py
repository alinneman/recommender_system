import torch
from torch.utils.data import Dataset

# Dataset with Price and Precomputed Embeddings
class RecommenderDataset(Dataset):
    def __init__(self, item_embeddings, prices, user_histories, pos_items, neg_items):
        self.N_MOST_RECENT = 40
        self.item_embeddings = item_embeddings
        self.prices = torch.tensor(prices, dtype=torch.float32)
        self.user_histories = user_histories
        self.pos_items = pos_items
        self.neg_items = neg_items
    
    def __len__(self):
        return len(self.user_histories)
    
    def __getitem__(self, idx):
        user_hist = self.user_histories[idx]
        pos_item = self.pos_items[idx]
        neg_item = self.neg_items[idx]
        user_price = self.prices[idx]
        return torch.tensor(user_hist), torch.tensor(pos_item), torch.tensor(neg_item), user_price
    
    
    def collate_fn(self, batch):
        user_histories, pos_items, neg_items, prices = zip(*batch)
        padded_histories = []
        for history in user_histories:
            if len(history) >= self.N_MOST_RECENT:
                padded_histories.append(history[-self.N_MOST_RECENT:])
            else:
                padded_histories.append(history + [0] * (self.N_MOST_RECENT - len(history)))
        return torch.tensor(padded_histories), torch.tensor(pos_items), torch.tensor(neg_items), torch.tensor(prices).unsqueeze(1)
    
    