import torch
import torch.nn as nn
import pytorch_lightning as pl

from .networks import ItemTower, UserTower
from .constants import TAU

# Two-Tower Model
class TwoTowerModel(pl.LightningModule):
    def __init__(self, vocab_size, category_size, learning_rate=1e-3):
        super(TwoTowerModel, self).__init__()
        self.item_tower = ItemTower(vocab_size, category_size)
        self.user_tower = UserTower(vocab_size, category_size)
        self.loss_fn = ContrastiveLoss()
        self.learning_rate = learning_rate
    
    def forward(self, user_items, user_cats, pos_items, pos_cats, neg_items, neg_cats):
        user_emb = self.user_tower(user_items, user_cats)
        pos_emb = self.item_tower(pos_items, pos_cats)
        neg_emb = self.item_tower(neg_items, neg_cats)
        pos_score = (user_emb * pos_emb).sum(dim=1) / TAU
        neg_score = (user_emb * neg_emb).sum(dim=1) / TAU
        return pos_score, neg_score

    def training_step(self, batch, batch_idx):
        user_items, user_cats, pos_items, pos_cats, neg_items, neg_cats = batch
        pos_score, neg_score = self.forward(user_items, user_cats, pos_items, pos_cats, neg_items, neg_cats)
        loss = self.loss_fn(pos_score, neg_score)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
