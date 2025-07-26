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
from exceptiongroup import catch
from tqdm import tqdm
import re
import time

from utils import extract_option, write_log_entry_to_csv, save_subject_logs


class MultiModelHandler:
    def __init__(self, num_model=None, eos_tokens=None, ports=None, max_workers=10,
                 enable_removal=False, removal_threshold=0.3, window_size=5, consecutive_windows=3, max_removals=2):
        self.file_path = "./model_info.json"
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.save_dir = "./out/saved_logs"
        os.makedirs(self.save_dir, exist_ok=True)

        # CEval数据集路径
        self.ceval_val_path = "./dataset/ceval-exam/val"
        self.ceval_results_dir = "./out/ceval_results"
        os.makedirs(self.ceval_results_dir, exist_ok=True)

        # 剔除机制参数
        self.enable_removal = enable_removal
        self.removal_threshold = removal_threshold
        self.window_size = window_size
        self.consecutive_windows = consecutive_windows
        self.max_removals = max_removals

        # 剔除状态跟踪
        self.removal_events = []
        self.all_problem_records = []
        self.model_macs_history = {}
        self.removal_count = 0
        self.current_models = []
        self.accuracy_history = []

    def generate_response(self, model_choice, question, args, problem_id=None, subject=None):
        import utils

        val, info = utils.validate_args(args)
        if not val:
            return info

        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 更新当前模型列表
        self.current_models = [data[i]["model_name"] for i in model_choice]

        # 初始化MACS历史记录
        for model_name in self.current_models:
            if model_name not in self.model_macs_history:
                self.model_macs_history[model_name] = {
                    'scores': [],
                    'low_score_count': 0,
                    'removed': False
                }

        self.models_info = [data[i] for i in model_choice]
        self.eos_tokens = [info['EOS'] for info in self.models_info]

        texts = [self.call_template(question, info) for info in self.models_info]

        new_word = ""
        len_of_tokens = 0
        ans = ""

        log_data = {
            "question": question,
            "model_names": [info["model_name"] for info in self.models_info],
            "steps": []
        }

        # 存储当前问题中各模型的MACS分数
        current_problem_macs = {model_name: [] for model_name in self.current_models}

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
                    topk = sorted(model_output['sample_result'], key=lambda x: x[1], reverse=True)[:args['top_k']]
                elif args['handel_next_token']:
                    pred_vals = model_output['prediction_values']
                    topk = sorted(pred_vals, key=lambda x: x[1], reverse=True)[:args['top_k']]
                else:
                    topk = sorted(model_output['sample_result'], key=lambda x: x[1], reverse=True)[:args['top_k']]

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

            # 计算每个模型的MACS贡献分数
            for model_data in return_args:
                self.calculate_macs(model_data, new_word, args['top_k'])
                model_name = model_data['model_name']
                current_problem_macs[model_name].append(model_data['c_total'])

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

        # 记录当前问题的MACS数据
        if problem_id and subject:
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
            self.all_problem_records.append(problem_macs_record)

        return ans, log_data, current_problem_macs

    def check_removal_condition(self, current_problem_macs, problem_id, subject):
        """检查是否满足剔除条件"""
        if not self.enable_removal or self.removal_count >= self.max_removals:
            return None

        models_to_remove = []

        for model_name, macs_scores in current_problem_macs.items():
            if self.model_macs_history[model_name]['removed']:
                continue

            if macs_scores:
                avg_macs = sum(macs_scores) / len(macs_scores)
            else:
                avg_macs = 0

            self.model_macs_history[model_name]['scores'].append(avg_macs)

            # 只保留最近的窗口大小
            if len(self.model_macs_history[model_name]['scores']) > self.window_size:
                self.model_macs_history[model_name]['scores'].pop(0)

            # 检查连续低于阈值的次数
            if len(self.model_macs_history[model_name]['scores']) >= self.consecutive_windows:
                last_n_scores = self.model_macs_history[model_name]['scores'][-self.consecutive_windows:]
                if all(score < self.removal_threshold for score in last_n_scores):
                    models_to_remove.append(model_name)
                    self.model_macs_history[model_name]['removed'] = True
                    self.removal_count += 1

        removal_events = []
        for model_name in models_to_remove:
            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "subject": subject,
                "problem_id": problem_id,
                "model_name": model_name,
                "window_scores": self.model_macs_history[model_name]['scores'][-self.consecutive_windows:],
                "threshold": self.removal_threshold,
                "remaining_models": [m for m in self.current_models if m != model_name]
            }
            self.removal_events.append(event)
            removal_events.append(event)
            print(f"⚠️ 模型 {model_name} 在问题 {problem_id} (科目: {subject}) 被剔除")

        return removal_events

    def call_app(self, text, info, extra_args):
        url = f"{info['model_url']}/predict"
        headers = {'Content-Type': 'application/json'}
        data = {'text': text, 'args': extra_args}
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

        if response.status_code == 200:
            result = response.json()
            return info["model_name"], result, elapsed_ms, model_eos_token
        else:
            print(f"{info['model_name']}, Error: {response.status_code}")
            return info["model_name"], {}, elapsed_ms, model_eos_token

    def evaluate_ceval(self, model_choice, args, subjects=None, max_samples=None):
        """
        在CEval验证集上评估模型性能，并添加剔除机制
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            models_data = json.load(file)

        model_names = [models_data[i]["model_name"] for i in model_choice]

        # 重置状态
        self.removal_events = []
        self.all_problem_records = []
        self.model_macs_history = {name: {'scores': [], 'low_score_count': 0, 'removed': False}
                                   for name in model_names}
        self.removal_count = 0
        self.current_models = model_names.copy()
        self.accuracy_history = []

        if subjects is None:
            subjects = [d for d in os.listdir(self.ceval_val_path)
                        if os.path.isdir(os.path.join(self.ceval_val_path, d))]

        all_results = {}
        total_correct = 0
        total_samples = 0
        cumulative_correct = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.ceval_results_dir, f"ceval_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # 初始化剔除机制
        active_model_choice = model_choice.copy()

        for subject in tqdm(subjects, desc="Subjects"):
            csv_path = os.path.join(self.ceval_val_path, subject) + '.csv'
            if not os.path.exists(csv_path):
                print(f"跳过 {subject} - CSV文件不存在: {csv_path}")
                continue

            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if all(field in row for field in ['id', 'question', 'A', 'B', 'C', 'D', 'answer']):
                        data.append(row)

            if not data:
                print(f"跳过 {subject} - CSV文件中无有效数据")
                continue

            if max_samples and max_samples < len(data):
                data = data[:max_samples]

            subject_correct = 0
            subject_results = []
            subject_log_data = []

            # 科目内的问题循环
            for i, item in enumerate(tqdm(data, desc=f"处理 {subject}", leave=False)):
                if not active_model_choice:
                    print("所有模型已被剔除，终止评估")
                    break

                # 构建问题
                question = self.format_ceval_question(item, subject)

                # 生成答案
                generated_answer, log_data, current_problem_macs = self.generate_response(
                    active_model_choice, question, args, problem_id=item["id"], subject=subject
                )

                # 检查剔除条件
                removal_events = self.check_removal_condition(
                    current_problem_macs, problem_id=item["id"], subject=subject
                )

                # 如果有模型被剔除，更新活动模型列表
                if removal_events:
                    for event in removal_events:
                        model_name = event["model_name"]
                        # 找到被剔除模型的索引
                        model_index = next(
                            (idx for idx, model_idx in enumerate(active_model_choice)
                             if models_data[model_idx]["model_name"] == model_name
                             ), None)
                        if model_index is not None:
                            active_model_choice.pop(model_index)

                # 提取选项并检查正确性
                predicted_option = extract_option(generated_answer)
                is_correct = predicted_option == item["answer"].upper()

                # 更新准确率
                if is_correct:
                    subject_correct += 1
                    total_correct += 1
                    cumulative_correct += 1


                total_samples += 1
                current_accuracy = cumulative_correct / total_samples * 100
                self.accuracy_history.append({
                    "subject": subject,
                    "problem_id": item["id"],
                    "accuracy": current_accuracy,
                    "models": [models_data[idx]["model_name"] for idx in active_model_choice]
                })

                # 记录结果
                result_entry = {
                    "id": item["id"],
                    "question": item["question"],
                    "options": {
                        "A": item["A"],
                        "B": item["B"],
                        "C": item["C"],
                        "D": item["D"]
                    },
                    "answer": item["answer"],
                    "generated_text": generated_answer,
                    "predicted_option": predicted_option,
                    "is_correct": is_correct,
                    "current_models": [models_data[idx]["model_name"] for idx in active_model_choice]
                }
                subject_results.append(result_entry)
                subject_log_data.append(log_data)

                # 记录问题详细信息
                problem_record = {
                    "subject": subject,
                    "problem_id": item["id"],
                    "is_correct": is_correct,
                    "accuracy_so_far": current_accuracy,
                    "models": [
                        {
                            "name": name,
                            "avg_macs": sum(scores) / len(scores) if scores else 0,
                            "scores": scores
                        }
                        for name, scores in current_problem_macs.items()
                    ],
                    "removal_occurred": bool(removal_events)
                }
                if removal_events:
                    problem_record["removal_events"] = removal_events
                self.all_problem_records.append(problem_record)
                time.sleep(1)

            # 计算科目准确率
            subject_acc = subject_correct / len(data) * 100 if len(data) > 0 else 0
            all_results[subject] = {
                "accuracy": subject_acc,
                "correct": subject_correct,
                "total": len(data),
                "results": subject_results
            }

            # 保存科目结果
            subject_file = os.path.join(result_dir, f"{subject}.json")
            with open(subject_file, 'w', encoding='utf-8') as f:
                json.dump(all_results[subject], f, indent=2, ensure_ascii=False)
            save_subject_logs(subject, subject_log_data, result_dir)

            # 保存剔除事件和问题记录
            self.save_removal_records(result_dir)

            if not active_model_choice:
                break


        time.sleep(30)
        # 计算总体准确率
        overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0

        # 保存总体结果
        summary = {
            "model_indexes": model_choice,
            "model_names": model_names,
            "subjects": list(all_results.keys()),
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "details": {subj: {"accuracy": all_results[subj]["accuracy"],
                               "correct": all_results[subj]["correct"],
                               "total": all_results[subj]["total"]}
                        for subj in all_results},
            "removal_events": self.removal_events,
            "final_models": [models_data[idx]["model_name"] for idx in active_model_choice]
        }

        summary_file = os.path.join(result_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 50)
        print(f"CEval评估完成 - 总体准确率: {overall_acc:.2f}%")
        print(f"模型剔除事件: {len(self.removal_events)}次")
        print("各科目准确率:")
        for subject, res in summary["details"].items():
            print(f"  {subject}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
        print("=" * 50)

        return overall_acc

    def save_removal_records(self, result_dir):
        """保存剔除记录和问题数据到文件"""
        # 保存剔除事件
        removal_file = os.path.join(result_dir, "removal_events.json")
        with open(removal_file, 'w', encoding='utf-8') as f:
            json.dump(self.removal_events, f, indent=2, ensure_ascii=False)

        # 保存所有问题记录到Excel
        if self.all_problem_records:
            df = pd.DataFrame(self.all_problem_records)

            model_data = []
            for _, row in df.iterrows():
                # 确保正确处理 models 字段，无论它是字典还是列表
                models = row['models']

                if isinstance(models, list):
                    # 当 models 是字典列表时
                    for model_info in models:
                        model_data.append({
                            "subject": row['subject'],
                            "problem_id": row['problem_id'],
                            "model_name": model_info.get('name', ''),
                            "avg_macs": model_info.get('avg_macs', 0),
                            "scores": json.dumps(model_info.get('scores', [])),
                            "is_correct": row['is_correct'],
                            "accuracy_so_far": row['accuracy_so_far'],
                            "removal_occurred": row['removal_occurred']
                        })
                elif isinstance(models, dict):
                    # 当 models 是模型名称到数据的映射时
                    for model_name, model_info in models.items():
                        model_data.append({
                            "subject": row['subject'],
                            "problem_id": row['problem_id'],
                            "model_name": model_name,
                            "avg_macs": model_info.get('avg_macs', 0),
                            "scores": json.dumps(model_info.get('scores', [])),
                            "is_correct": row['is_correct'],
                            "accuracy_so_far": row['accuracy_so_far'],
                            "removal_occurred": row['removal_occurred']
                        })
                else:
                    # 其他情况，打印警告
                    print(f"⚠️ 未知的models类型: {type(models)}")
                    continue

            model_df = pd.DataFrame(model_data)
            excel_file = os.path.join(result_dir, "all_problems_macs.xlsx")
            model_df.to_excel(excel_file, index=False)

        # 保存准确率历史
        if self.accuracy_history:
            accuracy_df = pd.DataFrame(self.accuracy_history)
            accuracy_file = os.path.join(result_dir, "accuracy_history.xlsx")
            accuracy_df.to_excel(accuracy_file, index=False)

    def format_ceval_question(self, item,subject):
        """
        格式化CEval问题为模型输入 - 修改为使用CSV格式
        """
        # 从CSV行中获取数据
        question = item["question"]
        options = {
            "A": item["A"],
            "B": item["B"],
            "C": item["C"],
            "D": item["D"]
        }

        # 构建选项字符串
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])

        # 获取科目信息（如果存在）
        subject = subject[:-4]

        return f"以下是一道{subject}的选择题，请直接输出答案选项（A、B、C或D）:\n\n{question}\n\n选项:\n{options_str}"



    def calculate_macs(self, model_data, selected_word, top_k):
        """
        计算模型的MACS贡献分数
        :param model_data: 模型数据字典
        :param selected_word: 最终选择的token
        :param top_k: 考虑的top-k数量
        """
        # 处理结束符映射
        target_token = selected_word
        if selected_word == '<end>':
            target_token = model_data['eos_token']

        # 在top-k tokens中查找目标token
        found_token = None
        for token_info in model_data['topk_token']:
            if token_info['token'] == target_token:
                found_token = token_info
                break

        # 计算logit投票支持度 (c_logit)
        top_tokens = model_data['topk_token']
        if top_tokens:
            max_logprob = top_tokens[0]['logprob']
            eps = 1e-10

            if found_token:
                logprob_y = found_token['logprob']
                c_logit = math.exp(logprob_y - max_logprob)
            else:
                c_logit = 0.0
        else:
            c_logit = 0.0

        # 计算Top-K命中情况 (c_rank)
        if found_token:
            rank = found_token['token_rank']
            c_rank = (top_k + 1 - rank) / top_k
        else:
            c_rank = 0.0

        # 计算最终贡献度 (加权组合)
        alpha = 0.5  # logit分量的权重
        beta = 0.5  # rank分量的权重
        c_total = alpha * c_logit + beta * c_rank

        # 将分数添加到模型数据中
        model_data['c_logit'] = c_logit
        model_data['c_rank'] = c_rank
        model_data['c_total'] = c_total

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


# 实例化处理器（启用剔除机制）
handler = MultiModelHandler(
    enable_removal=False,
    removal_threshold=0.3,  # MACS阈值
    window_size=5,  # 滑动窗口大小
    consecutive_windows=3,  # 连续低于阈值的窗口数
    max_removals=3  # 最多剔除模型的个数
)

# 模型组合列表
model_choice_list = [[0, 2, 3, 4]]

number_problems = 20
for model_choice in model_choice_list:
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
    subjects_to_evaluate_list = []
    # CEval评估（启用剔除机制）
    # subjects_to_evaluate = ['high_school_biology_val']
    subjects_to_evaluate = ["computer_network_val", "operating_system_val",
                            "computer_architecture_val",'high_school_biology_val',
                            'high_school_chemistry_val','high_school_chinese_val',
                            'high_school_geography_val','high_school_history_val',
                            'high_school_mathematics_val','high_school_physics_val',
                            'high_school_politics_val']
    # subjects_to_evaluate = ["computer_network_val", "operating_system_val",
    #                         "computer_architecture_val",'high_school_biology_val'
    #                        ]

    overall_acc = handler.evaluate_ceval(
        model_choice,
        args,
        subjects=subjects_to_evaluate,
        max_samples=number_problems
    )