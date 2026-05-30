import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def build_item_features(train_data: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    popularity = train_data['article_id'].value_counts().reset_index()
    popularity.columns = ['article_id', 'popularity']
    
    avg_price = train_data.groupby('article_id')['price'].mean().reset_index()
    avg_price.columns = ['article_id', 'price']

    item_features = pd.merge(articles[['article_id']], popularity, on='article_id', how='left')
    item_features = pd.merge(item_features, avg_price, on='article_id', how='left')
    
    item_features['popularity'] = item_features['popularity'].fillna(0)

    median_price = item_features['price'].median()
    item_features['price'] = item_features['price'].fillna(median_price)
    
    return item_features

def build_user_features(train_data: pd.DataFrame) -> pd.DataFrame:
    user_features = train_data.groupby('customer_id').agg(
        user_avg_spend=('price', 'mean')
    ).reset_index()
    return user_features

def generate_bert_embeddings(
    articles: pd.DataFrame, 
    text_col: str = 'detail_desc', 
    model_name: str = "cointegrated/rubert-tiny2", 
    batch_size: int = 128
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    texts = articles[text_col].fillna("No description").tolist()
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT Embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            batch_embeddings = model_output.last_hidden_state.mean(dim=1).cpu().numpy()
            
        embeddings.append(batch_embeddings)
        
    return np.vstack(embeddings)

import os
import json

def build_cold_start_artifacts(
    train_data: pd.DataFrame, 
    articles: pd.DataFrame, 
    customers: pd.DataFrame, 
    artifacts_dir: str
):
    os.makedirs(artifacts_dir, exist_ok=True)
    
    global_top = train_data['article_id'].value_counts().head(20).index.tolist()
    
    with open(os.path.join(artifacts_dir, 'global_top.json'), 'w', encoding='utf-8') as f:
        json.dump(global_top, f)
        
    df = train_data.merge(customers[['customer_id', 'age_group']], on='customer_id', how='left')
    df = df.merge(articles[['article_id', 'item_type']], on='article_id', how='left')
    
    demographic_tops = {}
    age_groups = ['16-24', '25-34', '35-49', '50+']
    
    for age in age_groups:
        age_data = df[df['age_group'] == age]
        
        if age_data.empty:
            demographic_tops[age] = {"Regularly": global_top, "NONE": global_top}
            continue
            
        pop_by_age = age_data['article_id'].value_counts().reset_index()
        pop_by_age.columns = ['article_id', 'count']
        pop_by_age = pop_by_age.merge(articles[['article_id', 'item_type']], on='article_id', how='left')
        
        fashion_top = pop_by_age[pop_by_age['item_type'] == 'Fashion']['article_id'].tolist()
        basic_top = pop_by_age[pop_by_age['item_type'] == 'Basic']['article_id'].tolist()
        
        regularly_top = (fashion_top + basic_top)[:20]
        
        none_top = (basic_top + fashion_top)[:20]
        
        if len(regularly_top) < 20:
            regularly_top = (regularly_top + [x for x in global_top if x not in regularly_top])[:20]
        if len(none_top) < 20:
            none_top = (none_top + [x for x in global_top if x not in none_top])[:20]
            
        demographic_tops[age] = {
            "Regularly": regularly_top,
            "NONE": none_top
        }
        
    with open(os.path.join(artifacts_dir, 'demographic_tops.json'), 'w', encoding='utf-8') as f:
        json.dump(demographic_tops, f)