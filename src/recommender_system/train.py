import datetime as dt
import json
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets.dataset import SessionDataset
from .models.networks import ItemTower, UserTower, TwoTowerRecommender
from .models.lightning import SessionRecommenderModule

def main():
    
    current_time = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = os.path.join('..', 'data')
    training_data_dir = os.path.join(data_dir, 'amazon_kdd')
    log_dir = os.path.join(data_dir, 'experiments', current_time)
    save_dir = os.path.join(log_dir, 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    pretrained_columns = ['brand', 'color', 'desc', 'material', 'model', 'size', 'title']
    cat_cols = ['emb_author', 'emb_price_bin']

    text_dict = {}
    for column in tqdm(pretrained_columns):
        file_name = os.path.join(training_data_dir, f'products_{column}_emb.parquet')
        temp_df = pd.read_parquet(file_name)
        text_dict[column] = temp_df.set_index(column)[f'{column}_emb'].to_dict()
    
    cat_emb_df = pd.read_parquet(os.path.join(training_data_dir, 'products_cat.parquet'))
    cat_emb_dict = cat_emb_df.set_index('id').to_dict('index')
    
    products_df = pd.read_csv(os.path.join(training_data_dir, 'products.parquet'))
    train_sessions_df = pd.read_parquet(os.path.join(training_data_dir, 'sessions_train.parquet'))
    val_sessions_df = pd.read_parquet(os.path.join(training_data_dir, 'sessions_val.parquet'))
    
    with open(os.path.join(training_data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    
    dataset_args = {
        'sessions_df': train_sessions_df,
        'item_df': products_df,
        'pretrained_emb_dict': text_dict,
        'cat_emb_dict': cat_emb_dict,
        'pretrained_cols': pretrained_columns,
        'cat_cols': cat_cols,
        'max_session_length': 40
    }
    
    train_dataset = SessionDataset(sessions_df=train_sessions_df, **dataset_args)
    val_dataset = SessionDataset(sessions_df=val_sessions_df, **dataset_args)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, collate_fn=val_dataset.collate_fn)
    
    # ---------------------------
    # Instantiate the item tower, user tower, and Two-Tower model.
    # ---------------------------
    
    item_tower = ItemTower(cat_vocab_size=metadata['vocab_size'], text_embedding_dim=384, cat_embedding_dim=192)
    # Create the user tower: input_dim matches item tower output (256).
    user_tower = UserTower(input_dim=256, gru_hidden_dim=256, mlp_hidden_dim=256, output_dim=256, dropout=0.1)
    
    model = TwoTowerRecommender(item_tower, user_tower, tau=0.1)
    
    # Wrap the model in the Lightning module.
    rec_module = SessionRecommenderModule(model, save_dir)
    
    # ---------------------------
    # Create PyTorch Lightning Trainer with early stopping and gradient clipping.
    # ---------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
    )
    tb_logger = TensorBoardLogger("lightning_logs/")
    trainer = pl.Trainer(
        max_epochs=50,
        gpus=1 if torch.cuda.is_available() else 0,
        callbacks=[early_stop_callback],
        gradient_clip_val=1.0,
        logger=tb_logger
    )
    
    # Train and validate.
    trainer.fit(rec_module, train_loader, val_loader)

    
    
if __name__ == "__main__":
    main()