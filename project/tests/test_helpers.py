import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
import pytest
 
from src.service.api import map_age_to_group, map_news_status
 
@pytest.mark.parametrize(
    "age,expected",
    [
        (20, "16-24"),
        (27, "25-34"),
        (40, "35-44"),
        (50, "45-54"),
        (60, "55-64"),
        (70, "65+"),
    ],
)
def test_map_age_to_group(age, expected):
    assert map_age_to_group(age) == expected


def test_map_age_to_group_none():
    assert map_age_to_group(None) is None


def test_map_age_boundaries():
    assert map_age_to_group(24) == "16-24"
    assert map_age_to_group(25) == "25-34"


@pytest.mark.parametrize(
    "has_news,expected",
    [
        (True, "Regularly"),
        (False, "NONE"),
        (None, None),
    ],
)
def test_map_news_status(has_news, expected):
    assert map_news_status(has_news) == expected
