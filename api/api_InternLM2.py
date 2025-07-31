from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os
import math
import uvicorn
from typing import Dict, Any, List
import torch
import sys
import importlib.util

app = FastAPI()

# 环境变量配置
PORT = int(os.getenv("PORT", 8000))
MODEL_PATH = os.getenv("MODEL_PATH", "/root/autodl-tmp/LLM/Shanghai_AI_Laboratory/internlm2-chat-7b")
MODEL_NAME = os.getenv("MODEL_NAME", "InternLM2-7B-Chat")
MODEL_ARCH = os.getenv("MODEL_ARCH", "transformers")
EOS_TOKEN = os.getenv("EOS_TOKEN", "<|im_end|>")
TEMPLATE_TYPE = os.getenv("TEMPLATE_TYPE", "internlm")

print(f"Initializing InternLM2 service for model: {MODEL_PATH}")

# 解决方案：手动加载InternLM2的tokenizer
try:
    # 方法1：尝试使用transformers的AutoTokenizer（如果版本支持）
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    print("Successfully loaded tokenizer via AutoTokenizer")
except Exception as e:
    print(f"AutoTokenizer failed: {str(e)}")
    print("Falling back to direct InternLM2Tokenizer import")

    try:
        # 方法2：直接导入InternLM2的tokenizer类
        sys.path.insert(0, MODEL_PATH)

        # 加载tokenization_internlm2模块
        tokenization_spec = importlib.util.spec_from_file_location(
            "tokenization_internlm2",
            os.path.join(MODEL_PATH, "tokenization_internlm2.py")
        )
        tokenization_module = importlib.util.module_from_spec(tokenization_spec)
        tokenization_spec.loader.exec_module(tokenization_module)

        # 创建tokenizer实例
        tokenizer = tokenization_module.InternLM2Tokenizer.from_pretrained(MODEL_PATH)
        print("Successfully loaded tokenizer via direct import")
    except Exception as e:
        print(f"Direct tokenizer import failed: {str(e)}")
        print("Attempting last-resort load with trust_remote_code")

        # 方法3：最后尝试强制使用trust_remote_code
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True
        )
        print("Successfully loaded tokenizer with forced trust_remote_code")

# 初始化vLLM引擎
print("Initializing vLLM engine...")
llm_args = {
    "model": MODEL_PATH,
    "tokenizer": tokenizer if 'tokenizer' in locals() else MODEL_PATH,
    "tensor_parallel_size": int(os.getenv("TENSOR_PARALLEL", 1)),
    "gpu_memory_utilization": float(os.getenv("GPU_MEM_UTIL", 0.9)),
    "trust_remote_code": True,
    "dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
    "max_model_len": 32768  # InternLM2支持长上下文
}

# 添加量化配置（如果启用）
if os.getenv("QUANTIZATION"):
    llm_args["quantization"] = os.getenv("QUANTIZATION")

llm = LLM(**llm_args)
print(f"Successfully loaded model: {MODEL_PATH} | Model name: {MODEL_NAME}")


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

        # 使用vLLM生成
        outputs = llm.generate([request.text], sampling_params)
        output = outputs[0]

        # 处理第一个生成的token
        token_id = output.outputs[0].token_ids[0]
        logprobs_dict = output.outputs[0].logprobs[0]

        # 提取实际token的logprob
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

        # 构建预测值列表
        prediction_values = [[item["token"], item["prob"]] for item in top_logprobs]

        # 构建样本结果
        sample_result = [[token_text, token_logprob]]

        # 合并参数
        response_args = {
            "model_name": MODEL_NAME,
            "model_arch": MODEL_ARCH,
            "eos_token": EOS_TOKEN,
            **request.args
        }

        # 构建响应
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/template")
async def template(request: TemplateRequest):
    try:
        question = request.question
        # InternLM2专用模板
        if TEMPLATE_TYPE == "internlm":
            formatted_text = (
                "<|im_start|>system\n"
                "You are an AI assistant whose name is InternLM (书生·浦语).\n"
                "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). "
                "It is designed to be helpful, honest, and harmless.\n"
                "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        # 其他模板类型保持不变...
        elif TEMPLATE_TYPE == "llama-chat":
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
    print(f"Starting InternLM2 API server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)