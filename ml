# models/base.py
from pydantic import BaseModel
from abc import ABC, abstractmethod

class AnalysisRequest(BaseModel):
    ticker: str
    params: dict = {}

class BaseQuantModel(ABC):
    name: str
    description: str
    parameters_schema: dict  # JSON Schema

    @abstractmethod
    async def analyze(self, request: AnalysisRequest) -> dict:
        pass

# main.py (FastAPI)
import importlib
from fastapi import FastAPI
from sqlalchemy.orm import Session
# ... db models ...

app = FastAPI()

# 모델 레지스트리 로드 (시작 시 or API 호출 시)
async def load_all_models():
    models = {}
    for module in os.listdir("models"):
        if module.endswith(".py") and module != "base.py":
            mod = importlib.import_module(f"models.{module[:-3]}")
            for attr in dir(mod):
                cls = getattr(mod, attr)
                if isinstance(cls, type) and issubclass(cls, BaseQuantModel) and cls != BaseQuantModel:
                    instance = cls()
                    models[instance.name] = instance
    return models

@app.get("/models")
async def list_models():
    # DB에서 메타데이터 조회 + 실제 인스턴스 로드
    return {"models": [model.dict() for model in db_models]}

@app.post("/analyze/{model_name}")
async def run_analysis(model_name: str, request: AnalysisRequest):
    model = loaded_models.get(model_name)
    if not model:
        raise HTTPException(404, "Model not found")
    
    # Celery나 BackgroundTasks로 비동기 실행 (계산이 오래 걸릴 경우)
    result = await model.analyze(request)
    return result