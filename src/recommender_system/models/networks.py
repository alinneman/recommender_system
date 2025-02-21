
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .constants import EMBED_DIM, HIDDEN_DIM, TAU

# Item Tower Model
class ItemTower(nn.Module):
    def __init__(self, vocab_size, category_size):
        super(ItemTower, self).__init__()
        self.title_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.category_emb = nn.Embedding(category_size, EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.Linear(2 * EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, EMBED_DIM)
        )
    
    def forward(self, title_idx, category_idx):
        title_vec = self.title_emb(title_idx)
        category_vec = self.category_emb(category_idx)
        item_features = torch.cat([title_vec, category_vec], dim=1)
        return torch.nn.functional.normalize(self.mlp(item_features), p=2, dim=1)

# User Tower Model
class UserTower(nn.Module):
    def __init__(self, vocab_size, category_size):
        super(UserTower, self).__init__()
        self.item_tower = ItemTower(vocab_size, category_size)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, EMBED_DIM)
    
    def forward(self, user_items, user_cats):
        item_embeds = self.item_tower(user_items, user_cats)  # Shape: (batch, seq, embed_dim)
        _, hidden = self.gru(item_embeds)
        user_embedding = self.fc(hidden.squeeze(0))
        return torch.nn.functional.normalize(user_embedding, p=2, dim=1)


# Loss Function
class ContrastiveLoss(nn.Module):
    def forward(self, pos_score, neg_score):
        return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
