from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict_model import get_recommendations, load_artifacts
from src.data.make_dataset import load_features_data

from src.utils.config import load_config
config = load_config()

ARTIFACTS_DIR = config["paths"]["artifacts_dir"]
DATA_DIR = config["paths"]["data_dir"]

item_features = None
user_history = {}

try:
    features_result = load_features_data(DATA_DIR) 
    if isinstance(features_result, tuple):
        item_features = features_result[1] 
    else:
        item_features = features_result

    artifacts_result = load_artifacts(ARTIFACTS_DIR)
    if artifacts_result and len(artifacts_result) > 5:
        user_history = artifacts_result[5]
except Exception as exc:
    print(f"Ошибка загрузки! Причина: {exc}")


app = FastAPI(
    title="H&M Recommendation API",
    version="1.1.0",
    description=(
        "HTTP-сервис для двухэтапной рекомендательной системы (FAISS + LightGBM). "
        "Адаптирован под удобный формат входных данных (чистый возраст и статус подписки)."
    ),
    docs_url="/docs",
    redoc_url=None,
)


def map_age_to_group(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    if age < 25:
        return "16-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"


def map_news_status(has_news: Optional[bool]) -> Optional[str]:
    if has_news is None:
        return None
    return "Regularly" if has_news else "NONE"



class RecommendRequest(BaseModel):
    customer_id: Optional[str] = Field(None, description="ID пользователя (если есть)")
    age: Optional[int] = Field(27, ge=0, le=120, description="Возраст пользователя цифрами (например, 27)")
    has_fashion_news: Optional[bool] = Field(None, description="Статус подписки на новости: true (есть) / false (нет)")
    top_k: int = Field(config["api"]["default_top_k"], ge=1, le=100, description="Количество возвращаемых рекомендаций")


class RecommendResponse(BaseModel):
    message: str = Field(..., description="Статус операции")
    scenario: str = Field(..., description="Сценарий выдачи (Warm Start / Demographic / Global)")
    recommendations: list[str] = Field(..., description="Список ID рекомендованных товаров")
    latency_ms: float = Field(..., description="Время обработки запроса в миллисекундах")



@app.get("/health", tags=["System"])
def health_check() -> dict:
    return {"status": "ok", "service": "recsys-api"}


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(request: RecommendRequest) -> RecommendResponse:
    start = perf_counter()

    if item_features is None:
        raise HTTPException(
            status_code=500, 
            detail="Модели не загружены в память сервера."
        )

    age_group = map_age_to_group(request.age)
    fashion_news_frequency = map_news_status(request.has_fashion_news)

    user_profile = {
        "age_group": age_group,
        "fashion_news_frequency": fashion_news_frequency
    }

    try:
        recs = get_recommendations(
            user_id=request.customer_id,
            artifacts_dir=ARTIFACTS_DIR,
            item_features=item_features,
            user_profile=user_profile,
            top_k=request.top_k
        )

        if request.customer_id and request.customer_id in user_history and len(user_history[request.customer_id]) > 0:
            scenario = "Warm Start (FAISS + LightGBM)"
        elif any(v is not None for v in user_profile.values()):
            scenario = f"Demographic Cold Start (Age: {age_group}, News: {fashion_news_frequency})"
        else:
            scenario = "Global Cold Start"

    except Exception as exc:
        raise HTTPException(
            status_code=500, 
            detail=f"Ошибка генерации рекомендаций: {exc}"
        ) from exc

    latency_ms = (perf_counter() - start) * 1000.0

    return RecommendResponse(
        message="Рекомендации успешно сформированы.",
        scenario=scenario,
        recommendations=recs,
        latency_ms=latency_ms
    )