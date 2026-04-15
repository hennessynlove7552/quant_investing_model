"""
주가(또는 수익률) 예측 모델 통합 허브.

- QuantFormer (arXiv:2404.00424): 양자화 분포 + MSE
- THGNN (CIKM'22): 그래프 + BCE
- LSTM / GRU, Ridge / SVM / RF: 논문 §2 preliminaries 계열
- LinearMomentum: 단순 베이스라인
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

import pandas as pd

# 지연 import로 torch/sklearn 미설치 시에도 허브 메타데이터는 동작하게 함


class PredictionHubError(Exception):
    """prediction_hub 모듈의 기본 에러 클래스"""

    pass


def _safe_cast(value: Any, target_type: type, param_name: str) -> Any:
    """안전한 타입 변환 헬퍼"""
    if value is None:
        return None
    try:
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise PredictionHubError(
            f"파라미터 '{param_name}' 타입 변환 실패: {value} -> {target_type.__name__} ({e})"
        ) from e


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """DataFrame 유효성 검사"""
    if not isinstance(df, pd.DataFrame):
        raise PredictionHubError(f"{name}은 pd.DataFrame이어야 합니다. 현재: {type(df).__name__}")
    if df.empty:
        raise PredictionHubError(f"{name}이 비어있습니다.")
    if df.isnull().all().any():
        raise PredictionHubError(f"{name}에 모두 null인 컬럼이 있습니다.")


def _safe_import(module_name: str, attr_name: Optional[str] = None) -> Any:
    """안전한 모듈 import"""
    try:
        module = importlib.import_module(module_name)
        if attr_name:
            return getattr(module, attr_name)
        return module
    except ImportError as e:
        raise PredictionHubError(
            f"모듈 '{module_name}' 로드 실패. 설치 확인: pip install {module_name}. ({e})"
        ) from e
    except AttributeError as e:
        raise PredictionHubError(
            f"모듈 '{module_name}'에서 '{attr_name}' 찾을 수 없음. ({e})"
        ) from e


def prediction_model_ids() -> List[str]:
    return [
        "quantformer",
        "thgnn",
        "lstm",
        "gru",
        "ridge",
        "svm",
        "random_forest",
        "linear_momentum",
    ]


def prediction_model_descriptions() -> Dict[str, str]:
    return {
        "quantformer": "QuantFormer (arXiv:2404.00424): 선형 임베딩·무위치인코딩 Transformer, 양자화 라벨+MSE",
        "thgnn": "THGNN (CIKM'22): 상관 그래프 + Transformer + 이종 GAT + 랭킹 BCE",
        "lstm": "LSTM 회귀: 다음날 수익률 예측 (§2.2.2 맥락)",
        "gru": "GRU 회귀: 다음날 수익률 예측 (§2.2.3 맥락)",
        "ridge": "Ridge 회귀: 평탄화 시퀀스 특성 (고전 ML 베이스라인)",
        "svm": "SVR(RBF): 비선형 커널 회귀 (§2.2.1 SVM 맥락)",
        "random_forest": "RandomForest 회귀: 앙상블 트리 베이스라인",
        "linear_momentum": "선형 모멘텀: 최근 누적 수익률 랭킹 (규칙 베이스라인)",
    }


def run_stock_prediction(
    model_id: str,
    prices: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    **kwargs,
) -> dict:
    """
    model_id: prediction_model_ids() 중 하나
    prices: 종가 DataFrame (일별, 컬럼=티커)
    volume: 거래량 DataFrame (같은 인덱스/컬럼 정렬 권장). 없으면 QuantFormer/LSTM 등에서 proxy 사용.
    
    Raises:
        PredictionHubError: 지원하지 않는 모델, import 실패, 파라미터 타입 오류 등
    """
    # 입력 검증
    if not isinstance(model_id, str):
        raise PredictionHubError(f"model_id는 문자열이어야 합니다. 현재: {type(model_id).__name__}")
    
    _validate_dataframe(prices, "prices")
    
    if volume is not None:
        _validate_dataframe(volume, "volume")
        if not prices.index.equals(volume.index):
            raise PredictionHubError("prices와 volume의 인덱스가 일치하지 않습니다.")
        if not prices.columns.equals(volume.columns):
            raise PredictionHubError("prices와 volume의 컬럼이 일치하지 않습니다.")
    
    mid = model_id.strip().lower().replace(" ", "_")
    if mid not in prediction_model_ids():
        raise PredictionHubError(
            f"지원하지 않는 모델: '{model_id}'. "
            f"사용 가능: {', '.join(prediction_model_ids())}"
        )

    try:
        if mid == "linear_momentum":
            train_predict_linear_return = _safe_import("ml_predictors", "train_predict_linear_return")
            lookback = _safe_cast(kwargs.get("lookback", 5), int, "lookback")
            return train_predict_linear_return(prices, lookback=lookback)

        if mid in ("ridge", "svm", "random_forest"):
            train_predict_sklearn = _safe_import("ml_predictors", "train_predict_sklearn")
            mt = {"random_forest": "rf"}.get(mid, mid)
            lookback = _safe_cast(kwargs.get("lookback", 20), int, "lookback")
            train_ratio = _safe_cast(kwargs.get("train_ratio", 0.7), float, "train_ratio")
            seed = _safe_cast(kwargs.get("seed", 42), int, "seed")
            
            return train_predict_sklearn(
                prices,
                volume,
                model_type=mt,
                lookback=lookback,
                train_ratio=train_ratio,
                seed=seed,
            )

        if mid in ("lstm", "gru"):
            train_predict_rnn = _safe_import("ml_predictors", "train_predict_rnn")
            lookback = _safe_cast(kwargs.get("lookback", 20), int, "lookback")
            hidden = _safe_cast(kwargs.get("hidden", 32), int, "hidden")
            num_layers = _safe_cast(kwargs.get("num_layers", 1), int, "num_layers")
            epochs = _safe_cast(kwargs.get("epochs", 15), int, "epochs")
            lr = _safe_cast(kwargs.get("lr", 1e-3), float, "lr")
            train_ratio = _safe_cast(kwargs.get("train_ratio", 0.7), float, "train_ratio")
            seed = _safe_cast(kwargs.get("seed", 42), int, "seed")
            
            return train_predict_rnn(
                prices,
                volume,
                cell=mid,
                lookback=lookback,
                hidden=hidden,
                num_layers=num_layers,
                epochs=epochs,
                lr=lr,
                train_ratio=train_ratio,
                seed=seed,
                device=kwargs.get("device"),
            )

        if mid == "quantformer":
            train_and_predict_quantformer = _safe_import("quantformer", "train_and_predict_quantformer")
            seq_len = _safe_cast(kwargs.get("seq_len", kwargs.get("lookback", 20)), int, "seq_len")
            rho = _safe_cast(kwargs.get("rho", 3), int, "rho")
            phi = _safe_cast(kwargs.get("phi", 0.2), float, "phi")
            include_null = _safe_cast(kwargs.get("include_null", True), bool, "include_null")
            train_ratio = _safe_cast(kwargs.get("train_ratio", 0.7), float, "train_ratio")
            epochs = _safe_cast(kwargs.get("epochs", 20), int, "epochs")
            lr = _safe_cast(kwargs.get("lr", 1e-3), float, "lr")
            d_model = _safe_cast(kwargs.get("d_model", 16), int, "d_model")
            nhead = _safe_cast(kwargs.get("nhead", 4), int, "nhead")
            num_layers = _safe_cast(kwargs.get("num_layers", 2), int, "num_layers")
            seed = _safe_cast(kwargs.get("seed", 42), int, "seed")
            
            return train_and_predict_quantformer(
                prices,
                volume,
                seq_len=seq_len,
                rho=rho,
                phi=phi,
                include_null=include_null,
                train_ratio=train_ratio,
                epochs=epochs,
                lr=lr,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                seed=seed,
                device=kwargs.get("device"),
            )

        if mid == "thgnn":
            qim = _safe_import("quant_investing_model")
            thgnn_train_and_predict = getattr(qim, "thgnn_train_and_predict", None)
            if thgnn_train_and_predict is None:
                raise PredictionHubError(
                    "quant_investing_model에서 thgnn_train_and_predict를 찾을 수 없습니다."
                )
            
            lookback = _safe_cast(kwargs.get("lookback", 20), int, "lookback")
            corr_window = _safe_cast(kwargs.get("corr_window", 20), int, "corr_window")
            corr_threshold = _safe_cast(kwargs.get("corr_threshold", 0.6), float, "corr_threshold")
            train_ratio = _safe_cast(kwargs.get("train_ratio", 0.7), float, "train_ratio")
            epochs = _safe_cast(kwargs.get("epochs", 10), int, "epochs")
            lr = _safe_cast(kwargs.get("lr", 3e-4), float, "lr")
            weight_decay = _safe_cast(kwargs.get("weight_decay", 1e-4), float, "weight_decay")
            seed = _safe_cast(kwargs.get("seed", 42), int, "seed")
            top_k = kwargs.get("top_k")
            
            return thgnn_train_and_predict(
                prices,
                lookback=lookback,
                corr_window=corr_window,
                corr_threshold=corr_threshold,
                top_k=top_k,
                train_ratio=train_ratio,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                seed=seed,
                device=kwargs.get("device"),
            )

        raise RuntimeError("unreachable")
    
    except PredictionHubError:
        raise
    except Exception as e:
        raise PredictionHubError(
            f"모델 '{model_id}' 실행 중 예기치 않은 에러: {type(e).__name__}: {e}"
        ) from e
