
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os
import math
import uvicorn
import multiprocessing
from typing import List, Dict, Any
import torch


# 模型配置类
class ModelConfig:
    def __init__(self, name: str, path: str, port: int, template: str, eos_token: str,
                 gpu_id: int, trust_remote_code: bool = False):
        self.name = name
        self.path = path
        self.port = port
        self.template = template
        self.eos_token = eos_token
        self.gpu_id = gpu_id
        self.trust_remote_code = trust_remote_code
        self.llm = None


# 模型配置 - 每个模型分配不同的GPU
MODELS = {
    "qwen": ModelConfig(
        name="Qwen2.5-7B-Instruct",
        path="/root/autodl-tmp/qwen/Qwen2.5-7B-Instruct",
        port=8000,
        template="qwen",
        eos_token="<|im_end|>",
        gpu_id=0,  # 使用GPU 0
        trust_remote_code=False
    ),
    "glm4": ModelConfig(
        name="GLM4-9B",
        path="/root/autodl-tmp/ZhipuAI/glm-4-9b-chat",
        port=8001,
        template="glm4",
        eos_token="<|endofpiece|>",
        gpu_id=1,  # 使用GPU 1
        trust_remote_code=True
    ),
    # "internlm2": ModelConfig(
    #     name="InternLM2-7B-Chat",
    #     path="/root/autodl-tmp/internlm/InternLM2-7B-Chat",
    #     port=8002,
    #     template="internlm2",
    #     eos_token="<|im_end|>",
    #     gpu_id=2,  # 使用GPU 2
    #     trust_remote_code=True
    # )


}


class PredictRequest(BaseModel):
    text: str
    args: Dict[str, Any]

    model_config = {
        "protected_namespaces": ()
    }


class TemplateRequest(BaseModel):
    question: str

    model_config = {
        "protected_namespaces": ()
    }


class NewPredictResponse(BaseModel):
    model_name: str
    response: Dict[str, Any]

    model_config = {
        "protected_namespaces": ()
    }


def create_app(model_config: ModelConfig):
    app = FastAPI()

    # 设置当前进程使用的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config.gpu_id)
    torch.cuda.set_device(model_config.gpu_id)

    print(f"Initializing {model_config.name} on GPU {model_config.gpu_id}")

    # 初始化vLLM引擎
    model_config.llm = LLM(
        model=model_config.path,
        tokenizer=model_config.path,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", 0.9)),
        trust_remote_code=model_config.trust_remote_code
    )
    print(f"Loaded model: {model_config.name} at {model_config.path} on GPU {model_config.gpu_id}")

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

            outputs = model_config.llm.generate([request.text], sampling_params)
            output = outputs[0]
            token_id = output.outputs[0].token_ids[0]
            logprobs_dict = output.outputs[0].logprobs[0]

            token_logprob_obj = logprobs_dict[token_id]
            token_text = token_logprob_obj.decoded_token
            token_logprob = token_logprob_obj.logprob

            top_logprobs = []
            sorted_items = sorted(logprobs_dict.items(), key=lambda x: x[1].rank)
            for j, (tid, logprob_obj) in enumerate(sorted_items):
                if j >= sampling_params.logprobs:
                    break
                token_str = logprob_obj.decoded_token
                if not token_str.strip() and tid == token_id:
                    token_str = model_config.eos_token
                top_logprobs.append({
                    "token": token_str,
                    "logprob": logprob_obj.logprob,
                    "prob": math.exp(logprob_obj.logprob)
                })

            prediction_values = [[item["token"], item["prob"]] for item in top_logprobs]
            sample_result = [[token_text, token_logprob]]

            response_args = {
                "model_name": model_config.name,
                "model_arch": "transformers",
                "eos_token": model_config.eos_token,
                **request.args
            }

            response_data = {
                "args": response_args,
                "error": None,
                "prediction_values": prediction_values,
                "sample_result": sample_result
            }

            return NewPredictResponse(
                model_name=model_config.name,
                response=response_data
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/template")
    async def template(request: TemplateRequest):
        try:
            question = request.question
            template_type = model_config.template

            if template_type == "llama-chat":
                formatted_text = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{question} [/INST]"
            elif template_type == "qwen":
                formatted_text = (
                    "<|im_start|>system\n"
                    "You are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            elif template_type == "zephyr":
                formatted_text = f"<|user|>\n{question}</s>\n<|assistant|>"
            elif template_type == "mistral":
                formatted_text = f"[INST] {question} [/INST]"
            elif template_type == "glm4":
                formatted_text = (
                    "<|system|>\n"
                    "You are a helpful assistant.\n"
                    "<|user|>\n"
                    f"{question}\n"
                    "<|assistant|>\n"
                )
            elif template_type == "internlm2":
                formatted_text = (
                    "<|im_start|>system\n"
                    "You are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
            else:
                formatted_text = question

            return {"text": formatted_text}

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_server(model_key: str):
    config = MODELS[model_key]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    torch.cuda.set_device(config.gpu_id)

    app = create_app(config)
    print(f"Starting {config.name} server on port {config.port} using GPU {config.gpu_id}")
    uvicorn.run(app, host="0.0.0.0", port=config.port)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # ✅ 关键修复点

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs available")

    # 验证GPU配置
    for model_key, config in MODELS.items():
        if config.gpu_id >= num_gpus:
            raise ValueError(f"Model {model_key} configured for GPU {config.gpu_id} but only {num_gpus} GPUs available")

    # 启动所有模型服务
    processes = []
    for model_key in MODELS.keys():
        p = multiprocessing.Process(target=run_server, args=(model_key,))
        p.start()
        processes.append(p)
        print(f"Started {MODELS[model_key].name} on port {MODELS[model_key].port} using GPU {MODELS[model_key].gpu_id}")

    for p in processes:
        p.join()
