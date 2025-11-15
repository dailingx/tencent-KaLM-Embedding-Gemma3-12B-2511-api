from sentence_transformers import SentenceTransformer
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import uvicorn

# 全局模型变量
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时清理资源"""
    # 启动时加载模型
    global model
    model = SentenceTransformer(
        "./KaLM-Embedding-Gemma3-12B-2511",
        trust_remote_code=True,
        local_files_only=True,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",  # Optional
        },
    )
    model.max_seq_length = 512
    yield
    # 关闭时可以在这里添加清理逻辑（如果需要）


# 初始化 FastAPI 应用
app = FastAPI(lifespan=lifespan)


class EmbeddingRequest(BaseModel):
    sentences: List[str]
    prompt: str


class EmbeddingResponse(BaseModel):
    code: int
    message: str
    data: dict


@app.post("/embedding", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    """
    获取文本嵌入向量
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="模型未加载")
        
        if not request.sentences:
            raise HTTPException(status_code=400, detail="sentences 不能为空")
        
        embeddings = model.encode(
            request.sentences,
            prompt=request.prompt,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
        )
        
        # 将 numpy 数组转换为列表
        embeddings_list = embeddings.tolist()
        
        return EmbeddingResponse(
            code=200,
            message="success",
            data={
                "embeddings": embeddings_list,
                "shape": list(embeddings.shape)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "code": 200,
        "message": "success",
        "data": {
            "status": "healthy",
            "model_loaded": model is not None
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8012)
