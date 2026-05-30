import lightgbm as lgb
import optuna
import faiss
import numpy as np
import pandas as pd
import os
import json
import warnings

from src.utils.config import load_config

warnings.filterwarnings('ignore')

train_config = load_config("training.yaml")
base_params = train_config["lightgbm_params"]["base"]
tuning_params = train_config["lightgbm_params"]["tuning"]
final_settings = train_config["lightgbm_params"]["final"]

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    
    emb_normalized = embeddings.copy().astype('float32')
    faiss.normalize_L2(emb_normalized)
    
    index.add(emb_normalized)
    return index

def tune_and_train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, 
                        X_valid: pd.DataFrame, y_valid: pd.Series, 
                        n_trials: int = 20) -> lgb.Booster:

    def objective(trial):
        params = base_params
        params.update({
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100) 
        })

        train_data = lgb.Dataset(X_train, label=y_train, params={'feature_pre_filter': False}, free_raw_data=False)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, free_raw_data=False)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=tuning_params["num_boost_round"], 
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=tuning_params["stopping_rounds"], verbose=False)]
        )
        return model.best_score['valid_0']['auc']

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params

    final_params = base_params
    final_params.update(best_params)

    train_data_final = lgb.Dataset(X_train, label=y_train, params={'feature_pre_filter': False}, free_raw_data=False)
    valid_data_final = lgb.Dataset(X_valid, label=y_valid, reference=train_data_final, free_raw_data=False)

    final_model = lgb.train(
        final_params,
        train_data_final,
        num_boost_round=final_settings["num_boost_round"], 
        valid_sets=[valid_data_final],
        callbacks=[lgb.early_stopping(stopping_rounds=final_settings["stopping_rounds"], verbose=False)]
    )
    
    return final_model


def save_artifacts(model: lgb.Booster, index: faiss.IndexFlatIP, 
                   embeddings: np.ndarray, mapping_dicts: dict, 
                   user_history_dict: dict, artifacts_dir: str):
    os.makedirs(artifacts_dir, exist_ok=True)

    lgb_path = os.path.join(artifacts_dir, 'final_lightgbm_model.txt')
    model.save_model(lgb_path)
    
    faiss_path = os.path.join(artifacts_dir, 'faiss_index.bin')
    faiss.write_index(index, faiss_path)
    
    emb_path = os.path.join(artifacts_dir, 'item_embeddings.npy')
    np.save(emb_path, embeddings)

    safe_mapping = {
        'item_id_to_idx': {str(k): int(v) for k, v in mapping_dicts['item_id_to_idx'].items()},
        'idx_to_item_id': {str(k): str(v) for k, v in mapping_dicts['idx_to_item_id'].items()}
    }
    with open(os.path.join(artifacts_dir, 'mapping_dicts.json'), 'w', encoding='utf-8') as f:
        json.dump(safe_mapping, f)
        
    safe_history = {str(k): [int(x) for x in v] for k, v in user_history_dict.items()}
    with open(os.path.join(artifacts_dir, 'user_history_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(safe_history, f)