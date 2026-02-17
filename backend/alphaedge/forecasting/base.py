"""Base types for forecasting module."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastPoint:
    date: str
    predicted: float
    lower_bound: float
    upper_bound: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "predicted": round(self.predicted, 4),
            "lower_bound": round(self.lower_bound, 4),
            "upper_bound": round(self.upper_bound, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ForecastResult:
    model_name: str
    horizon: str
    current_price: float
    forecasts: list[ForecastPoint]
    predicted_return: float
    direction: str  # "up", "down", "flat"
    confidence: float
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "horizon": self.horizon,
            "current_price": round(self.current_price, 4),
            "forecasts": [f.to_dict() for f in self.forecasts],
            "predicted_return": round(self.predicted_return, 4),
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "metrics": {k: round(v, 6) if isinstance(v, float) else v for k, v in self.metrics.items()},
            "warnings": self.warnings,
        }


class BaseForecaster:
    """Base class for all forecasters."""

    name: str = "base"

    def fit(self, train_data: pd.DataFrame) -> None:
        raise NotImplementedError

    def predict(self, horizon_days: int, current_price: float) -> ForecastResult:
        raise NotImplementedError

    @staticmethod
    def _direction(predicted_return: float) -> str:
        if predicted_return > 0.005:
            return "up"
        elif predicted_return < -0.005:
            return "down"
        return "flat"
