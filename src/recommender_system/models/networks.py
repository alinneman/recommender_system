import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ItemTower(nn.Module):
    def __init__(self, cat_vocab_size, cat_embedding_dim=384):
        super().__init__()

        # Categorical embeddings.
        self.categorical_embedding = nn.Embedding(cat_vocab_size + 1, cat_embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, pre_embs, cat_embs):
        cat_embs = self.categorical_embedding(cat_embs)
        
        # Concatenate embeddings and pass through the MLP.
        concat_embeds = torch.cat([pre_embs, cat_embs], dim=1)
        item_embedding = self.mlp(concat_embeds)
        return nn.functional.normalize(item_embedding, p=2, dim=1)



class UserTower(nn.Module):
    def __init__(self, cat_vocab_size, cat_embedding_dim=384, gru_hidden_dim=768, output_dim=512):
        super().__init__()
        self.categorical_embedding = nn.Embedding(cat_vocab_size + 1, cat_embedding_dim)
        self.gru = nn.GRU(input_size=cat_vocab_size+384, hidden_size=gru_hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, pre_embs, cat_embs):
        cat_embs = self.categorical_embedding(cat_embs)
        # Concatenate embeddings and pass through the GRU.
        #pack padded sequence?
        
        # session_embeddings: (batch_size, seq_length, 256)
        _, hidden = self.gru(torch.cat([pre_embs, cat_embs], dim=1))  # hidden: (1, batch_size, gru_hidden_dim)
        hidden = hidden.squeeze(0)                # (batch_size, gru_hidden_dim)
        user_embedding = self.mlp(hidden)           # (batch_size, output_dim)
        user_embedding = nn.functional.normalize(user_embedding, p=2, dim=1)
        return user_embedding



class TwoTowerRecommender(nn.Module):
    def __init__(self, item_tower: ItemTower, user_tower: UserTower, tau=0.1):
        super().__init__()
        self.item_tower = item_tower
        self.user_tower = user_tower
        self.tau = tau

    def forward(self, session_pre_embs, session_cat_embs, target_pre_embs, target_cat_embs) -> torch.Tensor:
        # Get item embeddings from the item tower.
        session_item_embeddings = self.user_tower(session_pre_embs, session_cat_embs)  # (batch_size, seq_length, 256)
        target_item_embeddings = self.item_tower(target_pre_embs, target_cat_embs)  # (batch_size, 256)
        # Dot product (with temperature scaling) between user and target embeddings.
        logits = torch.matmul(session_item_embeddings, target_item_embeddings.t()) / self.tau  # (batch_size, batch_size)
        return logits

    def save_models(self, path:str, epoch:int):
        torch.save(self.item_tower.state_dict(), os.path.join(path, f"item_tower_{epoch}.pth"))
        torch.save(self.user_tower.state_dict(), os.path.join(path, f"user_tower_{epoch}.pth"))


# Loss Function
class ContrastiveLoss(nn.Module):
    def forward(self, pos_score, neg_score):
        return -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
