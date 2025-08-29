from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os
import math
import uvicorn
import torch
from typing import List, Dict, Any

# 指定使用GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
app = FastAPI()
# MODEL_PATH = "/root/autodl-tmp/LLM/ZhipuAI/glm-4-9b-chat"  # 替换为实际的GLM-4模型路径
MODEL_PATH = "/home/administrator/du/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# 环境变量配置
PORT = int(os.getenv("PORT", 8002))  # 使用不同端口避免冲突
MODEL_NAME = "DeepseekR1-7B"
MODEL_ARCH = "transformers"
EOS_TOKEN = "<｜end▁of▁sentence｜>"  # GLM-4的结束符
TEMPLATE_TYPE = "deepseekr1"
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 8192))

# 初始化vLLM引擎
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    tensor_parallel_size=1,  # 单GPU运行
    trust_remote_code=True,  # GLM需要此参数
    gpu_memory_utilization=0.4,
    max_model_len=MAX_MODEL_LEN,  # 限制最大序列长度
    dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16"
)

# print(f"Loaded GLM-4 model on GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"Model path: {MODEL_PATH}")


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
        top_p = args.get("top_p")
        if top_p is None:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=1,
            temperature=args.get("temperature", 0.7),
            top_k=args.get("top_k", 10),
            top_p=top_p,
            max_tokens=1,
            logprobs=args.get("top_k", 10),
            skip_special_tokens=False
        )

        outputs = llm.generate([request.text], sampling_params)
        output = outputs[0]

        # 处理第一个生成的token
        token_id = output.outputs[0].token_ids[0]
        logprobs_dict = output.outputs[0].logprobs[0]

        token_logprob_obj = logprobs_dict[token_id]
        token_text = token_logprob_obj.decoded_token
        token_logprob = token_logprob_obj.logprob

        # 提取top logprobs
        top_logprobs = []
        sorted_items = sorted(logprobs_dict.items(), key=lambda x: x[1].rank)
        for j, (tid, logprob_obj) in enumerate(sorted_items):
            if j >= sampling_params.logprobs:
                break
            token_str = logprob_obj.decoded_token
            if not token_str.strip() and tid == token_id:
                token_str = EOS_TOKEN
            top_logprobs.append({
                "token": token_str,
                "logprob": logprob_obj.logprob,
                "prob": math.exp(logprob_obj.logprob)
            })

        # 构建响应
        prediction_values = [[item["token"], item["prob"]] for item in top_logprobs]
        sample_result = [[token_text, token_logprob]]

        response_data = {
            "args": {
                "model_name": MODEL_NAME,
                "model_arch": MODEL_ARCH,
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
    try:
        question = request.question
        # GLM-4专用模板
        if TEMPLATE_TYPE == "deepseekr1":
            formatted_text = (
                "<|system|>\n"
                "You are an AI assistant\n"
                "<|user|>\n"
                f"{question}\n"
                "<|assistant|>\n"
            )
        else:
            formatted_text = question

        return {"text": formatted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1)