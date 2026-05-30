import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

from src.utils.config import load_config
from src.data.make_dataset import preprocess_and_save_data, load_features_data, time_based_split
from src.features.build_features import build_item_features, build_user_features, generate_bert_embeddings, build_cold_start_artifacts
from src.models.train_model import build_faiss_index, tune_and_train_lgbm, save_artifacts

warnings.filterwarnings('ignore')

config = load_config()
BASE_DIR = config["paths"]["base_dir"]
DATA_RAW_DIR = config["paths"]["raw_data_dir"]
DATA_DIR = config["paths"]["data_dir"]
ARTIFACTS_DIR = config["paths"]["artifacts_dir"]

train_config = load_config("training.yaml")

def prepare_lgbm_data(transactions, user_history_dict, item_embeddings, index, item_id_to_idx, idx_to_item_id, item_features, user_features, k_candidates=train_config["retrieval"]["k_candidates"]):
    users = transactions['customer_id'].unique()
    valid_users = [u for u in users if u in user_history_dict and len(user_history_dict[u]) > 0]
    
    query_embs = np.array([item_embeddings[user_history_dict[u]].mean(axis=0) for u in valid_users]).astype('float32')
    import faiss
    faiss.normalize_L2(query_embs)
    
    distances, indices = index.search(query_embs, k_candidates)
    
    actual_dict = transactions.groupby('customer_id')['article_id'].apply(set).to_dict()
    rows = []
    
    for i, user in enumerate(valid_users):
        actual_items = actual_dict.get(user, set())
        for rank, (idx, score) in enumerate(zip(indices[i], distances[i])):
            if idx in idx_to_item_id:
                item_id = idx_to_item_id[idx]
                rows.append({
                    'customer_id': user,
                    'article_id': item_id,
                    'faiss_score': float(score),
                    'faiss_rank': rank + 1,
                    'target': 1 if item_id in actual_items else 0
                })
                
    df = pd.DataFrame(rows)
    
    df = df.merge(item_features, on='article_id', how='left')
    df = df.merge(user_features, on='customer_id', how='left')
    df['price'] = df['price'].fillna(item_features['price'].median())
    df['popularity'] = df['popularity'].fillna(0)
    df['user_avg_spend'] = df['user_avg_spend'].fillna(item_features['price'].median())
    df['price_diff'] = df['price'] - df['user_avg_spend']
    
    return df[train_config["features"]], df['target']


def main():
    if not os.path.exists(os.path.join(DATA_DIR, 'transactions_features.csv')):
        preprocess_and_save_data(DATA_RAW_DIR, DATA_DIR)
    
    trans, articles, customers = load_features_data(DATA_DIR)
    train_data, val_data, test_data = time_based_split(trans)
    
    item_features = build_item_features(train_data, articles)
    user_features = build_user_features(train_data)
    

    unique_articles = articles['article_id'].unique()
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(unique_articles)}
    idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}
    mapping_dicts = {'item_id_to_idx': item_id_to_idx, 'idx_to_item_id': idx_to_item_id}
    
    item_embeddings = generate_bert_embeddings(articles, text_col='detail_desc')
    
    user_history = train_data.copy()
    user_history['item_idx'] = user_history['article_id'].map(item_id_to_idx)
    user_history = user_history.dropna(subset=['item_idx'])
    user_history['item_idx'] = user_history['item_idx'].astype(int)
    user_history_dict = user_history.groupby('customer_id')['item_idx'].apply(list).to_dict()
    
    index = build_faiss_index(item_embeddings)

    X_train, y_train = prepare_lgbm_data(train_data, user_history_dict, item_embeddings, index, item_id_to_idx, idx_to_item_id, item_features, user_features)
    X_valid, y_valid = prepare_lgbm_data(val_data, user_history_dict, item_embeddings, index, item_id_to_idx, idx_to_item_id, item_features, user_features)
    
    final_model = tune_and_train_lgbm(X_train, y_train, X_valid, y_valid, n_trials=10)
    
    build_cold_start_artifacts(train_data, articles, customers, ARTIFACTS_DIR)
    
    save_artifacts(final_model, index, item_embeddings, mapping_dicts, user_history_dict, ARTIFACTS_DIR)
    

if __name__ == "__main__":
    main()