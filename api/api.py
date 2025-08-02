from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os
import math
import uvicorn
from typing import List, Dict, Any, Optional
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
app = FastAPI()
MODEL_PATH = "/root/autodl-tmp/LLM/qwen/Qwen2.5-7B-Instruct"  # HuggingFace模型ID

# 环境变量配置
PORT = int(os.getenv("PORT", 8000))
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen2.5-7B-Instruct")
MODEL_ARCH = os.getenv("MODEL_ARCH", "transformers")
EOS_TOKEN = os.getenv("EOS_TOKEN", "<|im_end|>")
TEMPLATE_TYPE = os.getenv("TEMPLATE_TYPE", "qwen")

# 初始化vLLM引擎
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL", 1)),
    gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", 0.9)),
)

print(f"Loaded model: {MODEL_PATH}")


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
            best_of=1,
            temperature=args.get("temperature", 0.8),
            top_k=args.get("top_k", 5),
            top_p=top_p,
            max_tokens=500,
            logprobs=args.get("top_k", 5),  # 返回top_k个logprobs
            skip_special_tokens=False
        )

        # 使用vLLM生成
        outputs = llm.generate([request.text], sampling_params)
        output = outputs[0]

        # 处理第一个生成的token
        token_id = output.outputs[0].token_ids[0]
        logprobs_dict = output.outputs[0].logprobs[0]  # 获取字典

        # 从字典提取实际token的logprob
        token_logprob_obj = logprobs_dict[token_id]
        token_text = token_logprob_obj.decoded_token
        token_logprob = token_logprob_obj.logprob

        # 按rank排序提取top logprobs
        top_logprobs = []
        sorted_items = sorted(logprobs_dict.items(), key=lambda x: x[1].rank)
        for j, (tid, logprob_obj) in enumerate(sorted_items):
            if j >= sampling_params.logprobs:
                break
            token_str = logprob_obj.decoded_token
            # 处理空字节情况
            if not token_str.strip() and tid == token_id:
                token_str = EOS_TOKEN
            top_logprobs.append({
                "token": token_str,
                "logprob": logprob_obj.logprob,
                "prob": math.exp(logprob_obj.logprob)
            })

        # 构建预测值列表 (token + 概率)
        prediction_values = []
        for item in top_logprobs:
            prediction_values.append([item["token"], item["prob"]])

        # 构建样本结果 (token + logprob)
        sample_result = [[token_text, token_logprob]]

        # 合并参数
        response_args = {
            "model_name": MODEL_NAME,
            "model_arch": MODEL_ARCH,
            "eos_token": EOS_TOKEN,
            **request.args  # 包含所有前端传入的参数
        }

        # 构建符合前端要求的响应
        response_data = {
            "args": response_args,
            "error": None,
            "prediction_values": prediction_values,
            "sample_result": sample_result
        }

        return NewPredictResponse(
            model_name=MODEL_NAME,
            response=response_data
        )

    except Exception as e:
        import traceback
        traceback.print_exc()  # 打印完整错误
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/template")
async def template(request: TemplateRequest):
    try:
        question = request.question
        # 根据模型类型应用不同的模板
        if TEMPLATE_TYPE == "llama-chat":
            formatted_text = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        elif TEMPLATE_TYPE == "qwen":
            formatted_text = (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif TEMPLATE_TYPE == "zephyr":
            formatted_text = f"<|user|>\n{question}</s>\n<|assistant|>"
        elif TEMPLATE_TYPE == "mistral":
            formatted_text = f"[INST] {question} [/INST]"
        else:  # 默认无模板
            formatted_text = question

        return {"text": formatted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)