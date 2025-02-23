import numpy as np
import torch
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(self, sessions_df, item_df, pretrained_emb_dict, cat_emb_dict, pretrained_cols, cat_cols, max_session_length=40):
        
        self.sessions = sessions_df["prev_items"].tolist()
        self.sessions = [sessions[1:-1].replace("'", '').split() for sessions in self.sessions]
        self.next_items = sessions_df["next_item_id"].tolist()
        self.items = item_df.set_index("item_id").to_dict(orient="index")

        self.pretrained_cols = pretrained_cols
        self.pretrained_emb_dict = pretrained_emb_dict

        self.cat_cols = cat_cols
        self.cat_embs = cat_emb_dict

        self.max_session_length = max_session_length

            
    def __len__(self):
        return len(self.data)
    
    def _get_pretrained_emb(self, item_id: str):
        feature_embs = []
        for pretrained_col in self.pretrained_cols:
            col_val = self.items[item_id][pretrained_col]
            feature_emb = self.pretrained_emb_dict[pretrained_col].get(col_val, np.zeros(384, dtype=np.int8))
            feature_embs.append(torch.from_numpy(feature_emb))
        return torch.stack(feature_embs, dim=0).to(torch.int8)
    
    def _get_cat_emb(self, item_id: str):
        cat_vals = self.cat_embs[item_id]
        cat_embs = [cat_vals[cat_col] for cat_col in self.cat_cols]
        return torch.tensor(cat_embs, dtype=torch.int8)
    
    def __getitem__(self, idx):
        prev_ids = self.sessions[idx]
        target_id = self.next_items[idx]
        
        session_pre_embs = torch.stack([self._get_pretrained_emb(prev_id) for prev_id in prev_ids])
        session_cat_embs = torch.stack([self._get_cat_emb(prev_id) for prev_id in prev_ids])
        
        target_pre_emb = self._get_pretrained_emb(target_id)
        target_cat_emb = self._get_cat_emb(target_id)
        
        return session_pre_embs, session_cat_embs, target_pre_emb, target_cat_emb

    def collate_fn(self, batch):
        #pack padded sequence instead
        session_pre_embs, session_cat_embs, target_pre_emb, target_cat_emb = batch
        
        return session_meta_batch, target_meta_batch
    



