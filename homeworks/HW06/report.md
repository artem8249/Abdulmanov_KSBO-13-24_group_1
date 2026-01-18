# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-01.csv`
- Размер: (12000, 30)
- Целевая переменная: `target` ('0': 0.68, '1': 0.32)
- Признаки: числовые 

## 2. Protocol

- Разбиение: train/test: 0.75/0.25, `random_state`=42
- Подбор: CV на train 5 фолдов. Обучали на 4 фолдах, валидировали на 1 фолде, повторяли 5 раз. Выбор лучших параметров: По roc_auc по всем 5 фолдам


- Метрики: 
Accuracy — быстрая оценка общей доли верных ответов.
F1 — показывает баланс precision/recall, критичен при неравных классах.
ROC-AUC — оценивает качество ранжирования

## 3. Models

Опишите, какие модели сравнивали и какие гиперпараметры подбирали.

Минимум:

- DummyClassifier (baseline)
- LogisticRegression ("lr__C": 10.0, "lr__penalty": "l2", "lr__solver": "lbfgs")
- DecisionTreeClassifier("ccp_alpha": 0.0, "max_depth": null, "min_samples_leaf": 20)
- RandomForestClassifier("max_depth": null, "max_features": "sqrt",  "min_samples_leaf": 1)
- HistGradientBoosting("learning_rate": 0.1, "max_depth": null, "max_leaf_nodes": 63)
- StackingClassifier 

## 4. Results
HistGradientBoosting
    "accuracy": 0.9363333333333334,
    "f1": 0.8982418753329782,
    "roc_auc": 0.9746823421867858,
 Stacking
    "accuracy": 0.938,
    "f1": 0.9020021074815595,
    "roc_auc": 0.9745000253923113,
RandomForest
    "accuracy": 0.9346666666666666,
    "f1": 0.8943965517241379,
    "roc_auc": 0.9706312528566349,
DecisionTree
    "accuracy": 0.8633333333333333,
    "f1": 0.7848898216159497,
    "roc_auc": 0.9105332385353713,
LogReg(scaled)
    "accuracy": 0.8296666666666667,
    "f1": 0.7146845337800112,
    "roc_auc": 0.878905083540704,
Dummy(most_frequent)
    "accuracy": 0.6766666666666666,
    "f1": 0.0,
    "roc_auc": 0.5,
- Победитель HistGradientBoosting по roc_aucю. Roc_auc = 0.9746823421867858

## 5. Analysis

- Устойчивость: что будет, если поменять `random_state` (хотя бы 5 прогонов для 1-2 моделей) – кратко
- Ошибки: confusion matrix для лучшей модели находится в artifacts/figures/confusion_matrix.png модель хорошо прдсказывает оба класса
- Интерпретация: permutation importance (top-15) находится в artifacts/figures/permutation_importance(top-15).png. Модель использует 2-3 сильных признака, остальные — слабый сигнал или шум. Упрощение модели может улучшить обобщающую способность

## 6. Conclusion

Контроль сложности — ключ для деревьев: max_depth, min_samples_leaf и ccp_alpha борются с переобучением
RandomForest декоррелирует деревья через max_features и бутстрап, повышая устойчивость
Бустинг последовательно учится на ошибках — требует аккуратного подбора learning_rate и n_estimators, чтобы не переобучиться
train/test split строго вначале
test — финальная проверка
Стратификация и фиксированный seed — обеспечивают воспроизводимость и репрезентативность выборок
