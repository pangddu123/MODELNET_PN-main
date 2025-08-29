from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os
import math
import uvicorn
from typing import Dict, Any
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

app = FastAPI()
MODEL_PATH = "/home/administrator/du/model/LLM-Research/Meta-Llama-3___1-8B-Instruct"

# 环境变量配置
PORT = int(os.getenv("PORT", 8003))
MODEL_NAME = os.getenv("MODEL_NAME", "Llama-3-8B-Instruct")
MODEL_ARCH = os.getenv("MODEL_ARCH", "transformers")
EOS_TOKEN = os.getenv("EOS_TOKEN", "<|im_end|>")
TEMPLATE_TYPE = os.getenv("TEMPLATE_TYPE", "llama")  # 使用新模板类型
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", 8192))
# 初始化vLLM引擎
llm = LLM(
    model=MODEL_PATH,
    tokenizer=MODEL_PATH,
    tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL", 1)),
    max_model_len=MAX_MODEL_LEN,  # 限制最大序列长度
    gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", 0.4))
)

print(f"Loaded Qwen3-8B model: {MODEL_PATH}")


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

        outputs = llm.generate([request.text], sampling_params)
        output = outputs[0]
        token_id = output.outputs[0].token_ids[0]
        logprobs_dict = output.outputs[0].logprobs[0]

        token_logprob_obj = logprobs_dict[token_id]
        token_text = token_logprob_obj.decoded_token
        token_logprob = token_logprob_obj.logprob

        # 处理空token的特殊情况
        if not token_text.strip():
            token_text = EOS_TOKEN

        top_logprobs = []
        sorted_items = sorted(logprobs_dict.items(), key=lambda x: x[1].rank)
        for j, (tid, logprob_obj) in enumerate(sorted_items):
            if j >= sampling_params.logprobs:
                break
            token_str = logprob_obj.decoded_token
            # 处理空字节情况
            if not token_str.strip():
                token_str = EOS_TOKEN
            top_logprobs.append({
                "token": token_str,
                "logprob": logprob_obj.logprob,
                "prob": math.exp(logprob_obj.logprob)
            })

        prediction_values = []
        for item in top_logprobs:
            prediction_values.append([item["token"], item["prob"]])

        sample_result = [[token_text, token_logprob]]

        response_args = {
            "model_name": MODEL_NAME,
            "model_arch": MODEL_ARCH,
            "eos_token": EOS_TOKEN,
            **request.args
        }

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

        # if TEMPLATE_TYPE == "llama":
        #     formatted_text = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
        # else:  # 默认无模板
        #     formatted_text = question
        formatted_text = question
        return {"text": formatted_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)