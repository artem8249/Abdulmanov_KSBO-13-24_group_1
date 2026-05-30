from pathlib import Path
import yaml

def load_config(config_name: str = "config.yaml") -> dict:
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_path = base_dir / "configs" / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"файл не найден {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    if "paths" in config:
        config["paths"]["base_dir"] = base_dir
        
    return config