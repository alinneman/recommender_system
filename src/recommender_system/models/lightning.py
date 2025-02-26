import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl

from .networks import TwoTowerRecommender

class SessionRecommenderModule(pl.LightningModule):
    def __init__(self, model: TwoTowerRecommender, save_path: str):
        super().__init__()
        self.model = model
        self.save_path = save_path
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, session_meta_batch, target_meta_batch):
        return self.model(session_meta_batch, target_meta_batch)
    
    def training_step(self, batch, batch_idx):
        session_meta_batch, target_meta_batch = batch
        logits = self.model(session_meta_batch, target_meta_batch)
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        session_meta_batch, target_meta_batch = batch
        logits = self.model(session_meta_batch, target_meta_batch)
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        loss = self.criterion(logits, labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return {"val_loss": loss}
    
    def _compute_metrics(self, data_loader, k=10):
        self.model.eval()
        all_recall = []
        all_mrr = []
        all_ndcg = []
        for session_meta_batch, target_meta_batch in data_loader:
            logits = self.model(session_meta_batch, target_meta_batch)  # (batch, batch)
            batch_size = logits.size(0)
            _, indices = logits.sort(dim=1, descending=True)
            for i in range(batch_size):
                rank = (indices[i] == i).nonzero(as_tuple=False).item() + 1
                recall = 1 if rank <= k else 0
                all_recall.append(recall)
                all_mrr.append(1.0 / rank)
                all_ndcg.append(1.0 / math.log2(rank + 1))
        return np.mean(all_recall), np.mean(all_mrr), np.mean(all_ndcg)
        
    
    def on_validation_epoch_end(self):
        
        recall, mrr, ndcg = self._compute_metrics(val_loader, k=10)
        # Log metrics.
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_mrr", mrr, prog_bar=True)
        self.log("val_ndcg", ndcg, prog_bar=True)
        
        self.model.save_models(self.save_path, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        session_meta_batch, target_meta_batch = batch
        logits = self.model(session_meta_batch, target_meta_batch)
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        loss = self.criterion(logits, labels)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return {"test_loss": loss}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        return [optimizer], [scheduler]

