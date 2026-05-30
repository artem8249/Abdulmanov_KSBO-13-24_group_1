import pandas as pd
import numpy as np
import lightgbm as lgb
import faiss
import json
import os
import warnings

from src.utils.config import load_config
train_config = load_config("training.yaml")
infer_config = load_config("inference.yaml")

warnings.filterwarnings('ignore')

def load_artifacts(artifacts_dir: str):
    model_path = os.path.join(artifacts_dir, 'final_lightgbm_model.txt')
    model = lgb.Booster(model_file=model_path)
    
    index_path = os.path.join(artifacts_dir, 'faiss_index.bin')
    index = faiss.read_index(index_path)
    
    emb_path = os.path.join(artifacts_dir, 'item_embeddings.npy')
    embeddings = np.load(emb_path)
    
    map_path = os.path.join(artifacts_dir, 'mapping_dicts.json')
    with open(map_path, 'r', encoding='utf-8') as f:
        mapping_dicts = json.load(f)
        item_id_to_idx = mapping_dicts['item_id_to_idx'] 
        idx_to_item_id = {int(k): str(v) for k, v in mapping_dicts['idx_to_item_id'].items()}
        
    hist_path = os.path.join(artifacts_dir, 'user_history_dict.json')
    with open(hist_path, 'r', encoding='utf-8') as f:
        user_history_dict = json.load(f)
        
    demographic_path = os.path.join(artifacts_dir, 'demographic_tops.json')
    with open(demographic_path, 'r', encoding='utf-8') as f:
        demographic_tops = json.load(f)

    global_path = os.path.join(artifacts_dir, 'global_top.json')
    with open(global_path, 'r', encoding='utf-8') as f:
        global_top = json.load(f)
        
    return model, index, embeddings, item_id_to_idx, idx_to_item_id, user_history_dict, demographic_tops, global_top

def _clean_profile_value(value, default):
    if pd.isna(value) or value is None or str(value).strip().lower() == 'nan':
        return default
    return str(value)

def get_recommendations(
    user_id: str, 
    artifacts_dir: str, 
    item_features: pd.DataFrame, 
    user_profile: dict = None, 
    top_k: int = None
) -> list:
    model, index, embeddings, item_id_to_idx, idx_to_item_id, user_history, demographic_tops, global_top = load_artifacts(artifacts_dir)
    
    if top_k is None:
        top_k = infer_config["ranking"]["default_top_k"]

    if user_id in user_history and len(user_history[user_id]) > 0:
        
        user_items_indices = user_history[user_id]
        user_vector = embeddings[user_items_indices].mean(axis=0, keepdims=True).astype('float32')
        faiss.normalize_L2(user_vector)
        
        user_history_articles = [idx_to_item_id[idx] for idx in user_items_indices if idx in idx_to_item_id]
        
        
        user_avg_spend = item_features[item_features['article_id'].isin(user_history_articles)]['price'].mean()
        if pd.isna(user_avg_spend):
            user_avg_spend = item_features['price'].median()
        
        distances, indices = index.search(user_vector, infer_config["retrieval"]["k_candidates"])
        
        candidate_rows = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx in idx_to_item_id:
                candidate_rows.append({
                    'customer_id': user_id,
                    'article_id': idx_to_item_id[idx],
                    'faiss_score': float(score),
                    'faiss_rank': rank + 1
                })
                
        df_candidates = pd.DataFrame(candidate_rows)
        df_candidates = df_candidates.merge(item_features, on='article_id', how='left')
        
        df_candidates['user_avg_spend'] = user_avg_spend
        df_candidates['price'] = df_candidates['price'].fillna(item_features['price'].median())
        df_candidates['popularity'] = df_candidates['popularity'].fillna(0)
        df_candidates['price_diff'] = df_candidates['price'] - df_candidates['user_avg_spend']
        
        df_candidates['predict_prob'] = model.predict(df_candidates[train_config["features"]])
        
        df_final = df_candidates.sort_values('predict_prob', ascending=False).head(top_k)
        return df_final['article_id'].tolist()
        
    if user_profile is not None and any(not pd.isna(v) for v in user_profile.values()):
        age_group = _clean_profile_value(user_profile.get('age_group'), default=infer_config["cold_start"]["default_age_group"])
        news_freq = _clean_profile_value(user_profile.get('fashion_news_frequency'), default=infer_config["cold_start"]["default_news_freq"])
                
        cohort_tops = demographic_tops.get(age_group, {})
        recommendations = cohort_tops.get(news_freq, global_top)
        
        return recommendations[:top_k]

    return global_top[:top_k]