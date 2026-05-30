import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from src.utils.config import load_config
 
 
def test_main_config_loads():
    config = load_config()
    assert "paths" in config
    assert "api" in config


def test_api_config_keys():
    config = load_config()
    api = config["api"]
    assert "host" in api
    assert "port" in api
    assert "default_top_k" in api
    assert isinstance(api["port"], int)


def test_paths_config_keys():
    config = load_config()
    paths = config["paths"]
    assert "data_dir" in paths
    assert "artifacts_dir" in paths


def test_training_config_loads():
    train_config = load_config("training.yaml")
    assert "features" in train_config
    assert isinstance(train_config["features"], list)
    assert len(train_config["features"]) > 0
    assert "retrieval" in train_config
