import os
import math
import json
import time
import random
import requests
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

import utils
from MACSManager import MACSManager
from test import CEvalTester, BoolQTester, SimpleMathTester

# /mnt/Data/xao/modelnet/load_model/output

class MultiModelHandler:
    """多模型协作处理类"""

    def __init__(self, num_model=None, eos_tokens=None, ports=None, max_workers=10,
                 enable_removal=False, removal_threshold=0.3, window_size=5,
                 consecutive_windows=3, max_removals=2):
        self.file_path = "./model_info.json"
        self.save_dir = "./out/saved_logs"
        os.makedirs(self.save_dir, exist_ok=True)

        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        # 初始化MACS管理器
        self.macs_manager = MACSManager(
            enable_removal=enable_removal,
            removal_threshold=removal_threshold,
            window_size=window_size,
            consecutive_windows=consecutive_windows,
            max_removals=max_removals
        )

        # 生成过程监控参数
        self.max_repetition_count = 5  # 最大允许重复次数
        self.repetition_window = 3     # 重复检测窗口大小
        self.max_meaningless_tokens = 10  # 最大无意义token数量


        # 初始化测试器
        self.ceval_tester = CEvalTester(self)
        self.boolq_tester = BoolQTester(self)
        self.mmlu_tester = CEvalTester(self,ceval_val_path="./dataset/MMLU_ceval/data/val",
                 ceval_results_dir="./out/mmlu_results")
        self.simpleMath_tester = SimpleMathTester(self)

    def generate_response(self, model_choice, question, args, problem_id=None, subject=None):


        val, info = utils.validate_args(args)
        if not val:
            return info

        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 更新当前模型列表
        model_names = [data[i]["model_name"] for i in model_choice]

        # 初始化MACS管理器中的模型状态
        if self.macs_manager.enable_removal:
            self.macs_manager.initialize_models(model_names)

        self.models_info = [data[i] for i in model_choice]
        self.eos_tokens = [info['EOS'] for info in self.models_info]

        texts = [self.call_template(question, info) for info in self.models_info]

        new_word = ""
        len_of_tokens = 0
        ans = ""

        log_data = {
            "question": question,
            "model_names": model_names,
            "steps": []
        }

        # 存储当前问题中各模型的MACS分数
        current_problem_macs = {model_name: [] for model_name in
                                model_names} if self.macs_manager.enable_removal else {}

        while new_word not in ['<end>']:
            if len_of_tokens > args['max_len'] or any(token in new_word for token in self.eos_tokens if token):
                break

            len_of_tokens += 1
            next_words = {}
            return_args = []

            # 提交任务
            all_task = [
                self.executor.submit(self.call_app, texts[i], info, args)
                for i, info in enumerate(self.models_info)
            ]

            # 收集结果
            for future in as_completed(all_task):
                model_name, model_output, response_time, eos_token = future.result()
                info = next((m for m in self.models_info if m['model_name'] == model_name), {})
                model_arch = info.get("model_arch", "")

                if args['return_dict_in_generate'] is not True:
                    topk = sorted(model_output['response']['sample_result'], key=lambda x: x[1], reverse=True)[:args['top_k']]
                elif args['handel_next_token']:
                    pred_vals = model_output['response']['prediction_values']
                    topk = sorted(pred_vals, key=lambda x: x[1], reverse=True)[:args['top_k']]
                else:
                    topk = sorted(model_output['response']['sample_result'], key=lambda x: x[1], reverse=True)[:args['top_k']]

                topk_with_detail = []
                for idx, token_data in enumerate(topk):
                    if len(token_data) == 3:
                        token, prob, logprob = token_data
                    else:
                        token, prob = token_data
                        logprob = math.log(prob + 1e-10)
                    topk_with_detail.append({
                        "token": token,
                        "prob": prob,
                        "logprob": logprob,
                        "token_rank": idx + 1
                    })

                return_args.append({
                    "model_name": model_name,
                    "model_arch": model_arch,
                    "topk_token": topk_with_detail,
                    "raw_args": model_output.get("args", {}),
                    "response_time_ms": round(response_time, 2),
                    "eos_token": eos_token
                })

                if args['mode'] == 1:
                    topk = [[word, 1] for word, *_ in topk]

                if args.get('prefix', False):
                    topk = utils.filter_prefixes(topk)

                next_words[model_name] = topk

            new_word, score = self.calculate_scores(next_words, args)

            if args['log-info']:
                print(f"[Step {len_of_tokens}] new_word: {new_word} | next_words: {next_words}")

            # 计算每个模型的MACS贡献分数（如果启用）
            if self.macs_manager.enable_removal:
                for model_data in return_args:
                    c_total = self.macs_manager.calculate_macs(model_data, new_word, args['top_k'])
                    model_name = model_data['model_name']
                    if model_name in current_problem_macs:  # 确保模型在列表中
                        current_problem_macs[model_name].append(c_total)

            for i in range(len(texts)):
                texts[i] += new_word
            ans += new_word

            log_data['steps'].append({
                "step": len_of_tokens,
                "next_words": next_words,
                "selected_word": new_word,
                "score": score,
                "return_args": return_args,
                "current_texts": texts.copy(),
                "current_ans": ans
            })

        log_data["final_answer"] = ans

        # 记录当前问题的MACS数据（如果启用）
        if self.macs_manager.enable_removal and problem_id and subject:
            problem_macs_record = {
                "subject": subject,
                "problem_id": problem_id,
                "models": {}
            }
            for model_name, macs_scores in current_problem_macs.items():
                avg_macs = sum(macs_scores) / len(macs_scores) if macs_scores else 0
                problem_macs_record["models"][model_name] = {
                    "scores": macs_scores,
                    "avg_macs": avg_macs
                }
            self.macs_manager.all_problem_records.append(problem_macs_record)

        return ans, log_data, current_problem_macs

    def call_app(self, text, info, extra_args):
        url = f"{info['model_url']}/predict"
        headers = {'Content-Type': 'application/json'}
        data = {'text': text, 'args': extra_args}
        # url = 'http://127.0.0.1:8000/predict'
        # headers = {'Content-Type': 'application/json'}
        # data = {'args': {'do_sample': True, 'handel_next_token': True, 'log-info': True, 'max_len': 500, 'max_new_tokens': 10, 'mode': 0, 'output_scores': True, 'prefix': False, 'return_dict_in_generate': True, 'soft': True, 'temperature': 0.8, 'top_k': 5, 'top_p': None}, 'text': '<|im_start|>system You are a helpful assistant.<|im_end|><|im_start|>user以下是一道abstract_algebra_val的选择题，不输出其他任何内容，请直接输出答案选项（A、B、C或D）:The cyclic subgroup of Z_24 generated by 18 has order选项:A. 4B. 8C. 12D. 6<|im_end|><|im_start|>assistant'}

        start_time = time.time()
        response = requests.post(url, json=data, headers=headers)
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000

        # 获取模型的EOS token
        model_eos_token = info.get('EOS', '')

        if info['model_arch'] != "transformers":
            result = {'prediction_values': [], 'args': {}, 'sample_result': []}
            res_data = response.json()
            if res_data['response'][0]['bytes'] == []:
                res_data['response'][0]['token'] = model_eos_token
            result['sample_result'].append([
                res_data['response'][0]['token'],
                res_data['response'][0]['logprob']
            ])
            for item in res_data['response'][0]['top_logprobs']:
                token = item['token']
                logprob = item['logprob']
                prob = math.exp(logprob)
                if item['bytes'] == []:
                    token = model_eos_token
                result['prediction_values'].append([token, prob, logprob])
            return info["model_name"], result, elapsed_ms, model_eos_token

        if info['model_arch'] == "vllm":
            response = requests.post(f"{info['model_url']}/predict",
                                     json={"text": text, "args": extra_args})
            if response.status_code == 200:
                result = response.json()
                # 转换vLLM响应格式
                token_info = result['response'][0]
                sample_result = [
                    [token_info['token'], token_info['logprob']]
                ]
                prediction_values = []
                for lp in token_info['top_logprobs']:
                    prob = math.exp(lp['logprob'])
                    prediction_values.append([
                        lp['token'], prob, lp['logprob']
                    ])

                return {
                    "model_name": info["model_name"],
                    "result": {
                        "sample_result": sample_result,
                        "prediction_values": prediction_values,
                        "args": result.get("args", {})
                    },
                    "response_time": elapsed_ms,
                    "eos_token": info.get('EOS', '')
                }

        if response.status_code == 200:
            result = response.json()
            return info["model_name"], result, elapsed_ms, model_eos_token
        else:
            print(f"{info['model_name']}, Error: {response.status_code}")
            return info["model_name"], {}, elapsed_ms, model_eos_token

    def call_template(self, question, info):
        url = f"{info['model_url']}/template"
        headers = {'Content-Type': 'application/json'}
        data = {'question': question}
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('text', '')
        else:
            print(f"Error: {response.status_code}")
            return ""

    def calculate_scores(self, data, args):
        scores = {}
        for _, values in data.items():
            for word, score in values:
                if word in self.eos_tokens:
                    word = "<end>"
                scores[word] = scores.get(word, 0) + score
        if not scores:
            return "<end>", 1.0
        max_score = max(scores.values())
        highest = [w for w, s in scores.items() if s == max_score]
        return random.choice(highest), max_score


    def evaluate_ceval(self, model_choice, args, subjects=None, max_samples=None):
        """委托给CEvalTester执行评估"""
        return self.ceval_tester.evaluate(model_choice, args, subjects, max_samples)

    def evaluate_mmlu(self, model_choice, args, subjects=None, max_samples=None):
        """委托给CEvalTester执行评估"""
        return self.mmlu_tester.evaluate(model_choice, args, subjects, max_samples)

    def evaluate_boolq(self, model_choice, args, max_samples=1000):
        """委托给CEvalTester执行评估"""
        return self.boolq_tester.evaluate(model_choice, args, max_samples)
    def evaluate_simpleMath(self, model_choice, args, max_samples=1000):
        """委托给CEvalTester执行评估"""
        return self.simpleMath_tester.evaluate(model_choice, args, max_samples)

if __name__ == '__main__':
    # url = 'http://127.0.0.1:8000/predict'
    # headers = {'Content-Type': 'application/json'}
    # data = {
    #     'args': {'do_sample': True, 'handel_next_token': True, 'log-info': True, 'max_len': 500, 'max_new_tokens': 10,
    #              'mode': 0, 'output_scores': True, 'prefix': False, 'return_dict_in_generate': True, 'soft': True,
    #              'temperature': 0.8, 'top_k': 5, 'top_p': None},
    #     'text': '<|im_start|>system You are a helpful assistant.<|im_end|><|im_start|>user以下是一道abstract_algebra_val的选择题，不输出其他任何内容，请直接输出答案选项（A、B、C或D）:The cyclic subgroup of Z_24 generated by 18 has order选项:A. 4B. 8C. 12D. 6<|im_end|><|im_start|>assistant'}
    #
    # start_time = time.time()
    # response = requests.post(url, json=data, headers=headers)
    # print(response.json())
    # 实例化处理器（启用剔除机制）
    # 记录整体测试开始时间


    handler = MultiModelHandler(
        enable_removal=False,
        removal_threshold=0.3,  # MACS阈值
        window_size=5,  # 滑动窗口大小
        consecutive_windows=3,  # 连续低于阈值的窗口数
        max_removals=3  # 最多剔除模型的个数
    )
#6、8、
    # 模型组合列表
    # model_choice_list = [ [2, 3], [2, 6], [2, 8], [3, 6]]
    model_choice_list = [ [0,1]]
    # model_choice_list = [ [2, 6, 8], [3, 6, 8], [2, 3, 6, 8]]


    total_start_time = time.time()
    print(f"[总体日志] 测试开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}")

    # 记录日志到文件
    log_filename = f"evaluation_log_{int(total_start_time)}.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(f"### 评估测试开始 ###\n")
        log_file.write(f"开始时间戳: {total_start_time}\n")
        log_file.write(f"可读开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}\n")
        log_file.write(f"评估模型数量: {len(model_choice_list)}种组合\n\n")

    number_problems = 100000000
    number_subjects = 1

    for model_index, model_choice in enumerate(model_choice_list):
        args = {
            'max_len': 500,
            'top_k': 5,
            'prefix': False,
            'soft': True,
            'log-info': True,
            'do_sample': True,
            'max_new_tokens': 10,
            'temperature': 0.8,
            'return_dict_in_generate': True,
            # 'return_dict_in_generate':False,
            'output_scores': True,
            # 'output_scores': False,
            'top_p': None,
            'handel_next_token': True,
            'mode': 0
        }

        # 记录当前模型组合开始时间
        combo_start_time = time.time()
        combo_start_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(combo_start_time))
        print(f"\n[组合日志] 模型组合 {model_index + 1}/{len(model_choice_list)} 开始评估 ({model_choice})")
        print(f"[组合日志] 开始时间: {combo_start_str}")

        ceval_folder_path = "./dataset/ceval-exam/val"  # 替换为实际路径
        ceval_subjects_to_evaluate = [f for f in os.listdir(ceval_folder_path)
                     if f.endswith(".csv") and os.path.isfile(os.path.join(ceval_folder_path, f))]
        # ceval_subjects_to_evaluate = ceval_subjects_to_evaluate[:number_subjects]

        MMLU_folder_path = "./dataset/MMLU_ceval/data/val"  # 替换为实际路径
        MMLU_subjects_to_evaluate = [f for f in os.listdir(MMLU_folder_path)
                     if f.endswith(".csv") and os.path.isfile(os.path.join(MMLU_folder_path, f))]
        # MMLU_subjects_to_evaluate = MMLU_subjects_to_evaluate[:number_subjects]

        overall_acc = handler.evaluate_ceval(model_choice, args, ceval_subjects_to_evaluate, max_samples=number_problems)
        overall_acc = handler.evaluate_mmlu(model_choice, args, MMLU_subjects_to_evaluate, max_samples=number_problems)
        overall_acc = handler.evaluate_boolq(model_choice, args, max_samples=number_problems)
        # overall_acc = handler.evaluate_simpleMath(model_choice, args, max_samples=number_problems)
        # 记录当前模型组合耗时
        combo_elapsed = time.time() - combo_start_time
        print(f"[组合日志] 评估完成! 耗时: {combo_elapsed:.2f}秒")

        # 写入组合日志
        with open(log_filename, "a") as log_file:
            log_file.write(f"--- 模型组合 {model_index + 1} ---\n")
            log_file.write(f"组合配置: {model_choice}\n")
            log_file.write(f"开始时间: {combo_start_str}\n")
            log_file.write(f"评估耗时: {combo_elapsed:.2f}秒\n\n")
    # 记录整体测试结束时间和耗时
    total_end_time = time.time()
    total_elapsed = total_end_time - total_start_time
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_end_time))

    print(f"\n[总体日志] 测试结束时间: {time_str}")
    print(f"[总体日志] 总耗时: {total_elapsed:.2f}秒")

    # 更新日志文件
    with open(log_filename, "a") as log_file:
        log_file.write(f"### 评估测试结束 ###\n")
        log_file.write(f"结束时间戳: {total_end_time}\n")
        log_file.write(f"可读结束时间: {time_str}\n")
        log_file.write(f"总评估耗时: {total_elapsed:.2f}秒\n")
        log_file.write(f"平均每个模型组合耗时: {total_elapsed / len(model_choice_list):.2f}秒\n")
