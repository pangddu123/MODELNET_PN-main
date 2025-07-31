from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os
import math
import uvicorn
from typing import Dict, Any

app = FastAPI()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 环境变量配置（建议通过.env文件或容器环境设置）
MODEL_PATH = os.getenv("MODEL_PATH", "/root/autodl-tmp/LLM/qwen/Qwen1.5-7B-Chat")  # 修改为Qwen1.5模型路径
PORT = int(os.getenv("PORT", 8001))
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen1.5-7B-Chat")
TENSOR_PARALLEL = int(os.getenv("TENSOR_PARALLEL", 1))
GPU_MEM_UTIL = float(os.getenv("GPU_MEM_UTIL", 0.9))
EOS_TOKEN = os.getenv("EOS_TOKEN", "<|im_end|>")  # Qwen1.5使用相同的结束符
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 8192))

# 初始化vLLM引擎（添加trust_remote_code支持Qwen1.5）
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    tensor_parallel_size=TENSOR_PARALLEL,
    gpu_memory_utilization=GPU_MEM_UTIL,
    trust_remote_code=True,  # Qwen1.5需要此参数
    max_model_len=MAX_MODEL_LEN  # 限制最大序列长度
)

print(f"Loaded Qwen1.5 model: {MODEL_PATH}")


class PredictRequest(BaseModel):
    text: str
    args: Dict[str, Any]


class TemplateRequest(BaseModel):
    question: str


class NewPredictResponse(BaseModel):
    model_name: str
    response: Dict[str, Any]


@app.post("/predict", response_model=NewPredictResponse)
async def predict(request: PredictRequest):
    try:
        args = request.args
        sampling_params = SamplingParams(
            n=1,
            best_of=1,
            temperature=args.get("temperature", 0.8),
            top_k=args.get("top_k", 5),
            top_p=args.get("top_p", 1.0),
            max_tokens=1,
            logprobs=args.get("top_k", 5),
            skip_special_tokens=False
        )

        outputs = llm.generate([request.text], sampling_params)
        output = outputs[0]

        # 处理第一个token的输出
        token_id = output.outputs[0].token_ids[0]
        logprobs_dict = output.outputs[0].logprobs[0]
        token_logprob_obj = logprobs_dict[token_id]

        # 处理特殊token显示
        token_text = token_logprob_obj.decoded_token
        if not token_text.strip():
            token_text = EOS_TOKEN

        # 收集top logprobs
        top_logprobs = []
        for tid, logprob_obj in logprobs_dict.items():
            display_token = logprob_obj.decoded_token.strip() or EOS_TOKEN
            top_logprobs.append({
                "token": display_token,
                "logprob": logprob_obj.logprob,
                "prob": math.exp(logprob_obj.logprob)
            })
            if len(top_logprobs) >= sampling_params.logprobs:
                break

        # 构建响应结构
        prediction_values = [[item["token"], item["prob"]] for item in top_logprobs]
        sample_result = [[token_text, token_logprob_obj.logprob]]

        response_data = {
            "args": {
                "model_name": MODEL_NAME,
                "model_arch": "transformers",
                "eos_token": EOS_TOKEN,
                **request.args
            },
            "error": None,
            "prediction_values": prediction_values,
            "sample_result": sample_result
        }

        return NewPredictResponse(
            model_name=MODEL_NAME,
            response=response_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/template")
async def template(request: TemplateRequest):
    """Qwen1.5专用模板格式化"""
    try:
        # Qwen1.5官方对话模板
        formatted_text = (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{request.question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return {"text": formatted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)