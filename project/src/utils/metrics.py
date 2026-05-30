import numpy as np
from typing import Dict, List, Tuple

def calculate_metrics(
    actual_dict: Dict[str, List[str]], 
    predicted_dict: Dict[str, List[str]], 
    k: int = 20
) -> Tuple[float, float]:
    precisions = []
    recalls = []
    
    for user, actual_items in actual_dict.items():
        if user not in predicted_dict: 
            continue
            
        actual_set = set(actual_items)
        predicted_set = set(predicted_dict[user][:k])
        
        hits = len(actual_set & predicted_set)
        
        precisions.append(hits / k)
        recalls.append(hits / len(actual_set) if len(actual_set) > 0 else 0)
    
    if not precisions:
        return 0.0, 0.0
        
    return np.mean(precisions), np.mean(recalls)