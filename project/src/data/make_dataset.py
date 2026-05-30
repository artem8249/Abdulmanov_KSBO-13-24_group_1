import pandas as pd
import numpy as np
from pathlib import Path
import gc
from typing import Tuple

from src.utils.config import load_config
config = load_config()

def preprocess_and_save_data(data_dir: str, output_dir: str):
    raw_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    articles = pd.read_csv(raw_path / 'articles.csv', dtype={'article_id': str})
    customers = pd.read_csv(raw_path / 'customers.csv')
    trans_file = raw_path / 'transactions_train.csv'
    
    last_date = pd.read_csv(trans_file, usecols=['t_dat'])['t_dat'].max()
    cutoff = pd.to_datetime(last_date) - pd.Timedelta(days=config["preprocessing"]["cutoff_days"])
    chunks = pd.read_csv(trans_file, chunksize= config["preprocessing"]["chunk_size"], 
                         dtype={'article_id': str, 'price': 'float32'}, 
                         parse_dates=['t_dat'])
    transactions_mini = pd.concat([c[c['t_dat'] >= cutoff] for c in chunks])
    
    transactions_mini = transactions_mini.drop_duplicates()
    
    item_dates = pd.read_csv(trans_file, 
                             usecols=['article_id', 't_dat'], 
                             dtype={'article_id': str}, 
                             parse_dates=['t_dat'])
    
    lifecycle = item_dates.groupby('article_id')['t_dat'].agg(['min', 'max'])
    lifecycle.columns = ['first_sale', 'last_sale']
    one_month_ago = lifecycle['last_sale'].max() - pd.Timedelta(days=config["preprocessing"]["cutoff_days"])
    
    def classify_item_status(row):
        if row['last_sale'] < one_month_ago:
            return 'Archive'  
        if row['first_sale'] >= one_month_ago:
            return 'New'
        return 'Active'
        
    lifecycle['status'] = lifecycle.apply(classify_item_status, axis=1)
    del item_dates
    gc.collect()
    
    articles_with_status = articles.merge(lifecycle[['status']], on='article_id', how='left')
    articles_mini = articles_with_status[articles_with_status['status'].isin(['Active', 'New'])].copy()
    articles_mini['detail_desc'] = articles_mini['detail_desc'].fillna('No description')
    
    transactions_mini = transactions_mini[transactions_mini['article_id'].isin(articles_mini['article_id'])]
    
    basic_keywords = ['basic', 'pack', 'multipack', 'essential', 'tights', 'socks', 'underwear']
    
    def check_text_base(row):
        text = str(row['prod_name']).lower() + " " + str(row['detail_desc']).lower()
        return any(word in text for word in basic_keywords)
        
    quantity_per_order = transactions_mini.groupby(['customer_id', 'article_id', 't_dat']).size().reset_index(name='qty')
    avg_qty = quantity_per_order.groupby('article_id')['qty'].mean()
    
    articles_mini['text_is_basic'] = articles_mini.apply(check_text_base, axis=1)
    articles_mini['behavior_is_basic'] = articles_mini['article_id'].map(avg_qty) > 1.3
    
    def classify_item_type(row):
        if row['text_is_basic'] or row['behavior_is_basic']:
            return 'Basic'
        return 'Fashion'
        
    articles_mini['item_type'] = articles_mini.apply(classify_item_type, axis=1)
    articles_mini.drop(columns=['text_is_basic', 'behavior_is_basic'], inplace=True)
    
    item_prices = transactions_mini.groupby('article_id')['price'].median().reset_index()
    articles_mini = articles_mini.merge(item_prices, on='article_id', how='left')
    articles_mini['price'] = articles_mini['price'].fillna(articles_mini['price'].median())
    
    item_popularity = transactions_mini.groupby('article_id').size().reset_index(name='popularity')
    articles_mini = articles_mini.merge(item_popularity, on='article_id', how='left')
    articles_mini['popularity'] = articles_mini['popularity'].fillna(0)

    articles_mini['price_segment'] = pd.qcut(
        articles_mini['price'], 
        q=config["preprocessing"]["price_quantile_bins"], 
        labels=['Budget', 'Medium', 'Premium']
    )
    
    active_users = transactions_mini['customer_id'].unique()
    customers_mini = customers[customers['customer_id'].isin(active_users)].copy()
    
    bins = [15, 24, 34, 49, 100]
    labels = ['16-24', '25-34', '35-49', '50+']
    customers_mini['age_group'] = pd.cut(customers_mini['age'], bins=bins, labels=labels)
    customers_mini['fashion_news_frequency'] = customers_mini['fashion_news_frequency'].replace('None', 'NONE')
    
    transactions_mini.to_csv(out_path / 'transactions_features.csv', index=False)
    articles_mini.to_csv(out_path / 'articles_features.csv', index=False)
    customers_mini.to_csv(out_path / 'customers_features.csv', index=False)


def load_features_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    transactions = pd.read_csv(data_path / 'transactions_features.csv', parse_dates=['t_dat'], dtype={'article_id': str})
    articles = pd.read_csv(data_path / 'articles_features.csv', dtype={'article_id': str})
    customers = pd.read_csv(data_path / 'customers_features.csv')
    return transactions, articles, customers


def time_based_split(transactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_date = transactions['t_dat'].max()
    test_start = max_date - pd.Timedelta(days=config["preprocessing"]["days_for_val/test"])  
    val_start = test_start - pd.Timedelta(days=config["preprocessing"]["days_for_val/test"]) 
    
    test_data = transactions[transactions['t_dat'] > test_start].copy()
    val_data = transactions[(transactions['t_dat'] > val_start) & (transactions['t_dat'] <= test_start)].copy()
    train_data = transactions[transactions['t_dat'] <= val_start].copy()
    
    return train_data, val_data, test_data
