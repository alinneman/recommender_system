import pandas as pd

from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')

    products_df = pd.read_csv('/app/data/amazon_kdd/products_train.csv')
    products_df = products_df[products_df['locale'] == 'UK'].reset_index(drop=True)

    embed_cols = ['title','brand', 'color', 'size', 'model', 'material', 'author', 'desc']

    for embed_col in embed_cols:
        temp_df = products_df.copy()[['id', embed_col]]
        temp_df[f'{embed_col}_emb'] = model.encode(temp_df[embed_col].fillna(''), batch_size=256, precision='int8', show_progress_bar=True).tolist()
        temp_df.to_parquet(f'/app/data/amazon_kdd/products_{embed_col}_emb.parquet')

