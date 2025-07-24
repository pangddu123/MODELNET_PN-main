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
import re
import time  # 确保这行在文件顶部

class MultiModelHandler:
    def __init__(self, num_model=None, eos_tokens=None, ports=None, max_workers=10):
        self.file_path = "./model_info.json"
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.save_dir = "./out/saved_logs"
        os.makedirs(self.save_dir, exist_ok=True)

        # CEval数据集路径 - 请根据实际路径修改
        self.ceval_val_path = "./dataset/ceval-exam/val"
        self.ceval_results_dir = "./out/ceval_results"
        os.makedirs(self.ceval_results_dir, exist_ok=True)

    def generate_response(self, model_choice, question, args):
        import utils  # 确保 utils 是可访问模块

        val, info = utils.validate_args(args)
        if not val:
            return info

        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
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
                # model_url = info.get("model_url", "")

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
                    # "model_url": model_url,
                    "topk_token": topk_with_detail,
                    "raw_args": model_output.get("args", {}),
                    "response_time_ms": round(response_time, 2),
                    "eos_token": eos_token  # 保存每个模型的EOS token
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

        return ans,log_data

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
        在CEval验证集上评估模型性能
        :param model_choice: 选择的模型列表
        :param args: 生成参数
        :param subjects: 要评估的科目列表，如果为None则评估所有科目
        :param max_samples: 每个科目最大样本数，如果为None则使用全部
        :return: 总体准确率
        """

        # 加载模型信息文件
        with open(self.file_path, 'r', encoding='utf-8') as file:
            models_data = json.load(file)

        # 获取模型名称列表
        model_names = [models_data[i]["model_name"] for i in model_choice]


        # 加载科目列表
        if subjects is None:
            subjects = [d for d in os.listdir(self.ceval_val_path)
                        if os.path.isdir(os.path.join(self.ceval_val_path, d))]

        all_results = {}
        total_correct = 0
        total_samples = 0

        # 创建时间戳目录保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.ceval_results_dir, f"ceval_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # 遍历每个科目
        for subject in tqdm(subjects, desc="Subjects"):
            # 修改：使用CSV文件而不是JSON
            csv_path = os.path.join(self.ceval_val_path, subject) + '.csv'
            if not os.path.exists(csv_path):
                print(f"跳过 {subject} - CSV文件不存在: {csv_path}")
                continue

            # 加载CSV验证集
            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 确保所有字段都存在
                    if all(field in row for field in ['id', 'question', 'A', 'B', 'C', 'D', 'answer']):
                        data.append(row)

            if not data:
                print(f"跳过 {subject} - CSV文件中无有效数据")
                continue

            if max_samples > len(data):
                max_samples = len(data)
            # 限制样本数
            if max_samples is not None:
                data = data[:max_samples]

            subject_correct = 0
            subject_results = []

            subject_log_data = []
            # 处理每个问题
            for i, item in enumerate(tqdm(data, desc=f"处理 {subject}", leave=False)):
                # 构建问题
                question = self.format_ceval_question(item,subject)
                # 生成答案
                generated_answer, log_data = self.generate_response(model_choice, question, args)
                # 提取选项
                predicted_option = self.extract_option(generated_answer)
                # 检查是否正确
                is_correct = predicted_option == item["answer"].upper()

                # 记录MACS等数据
                subject_log_data.append(log_data)
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
                    "is_correct": is_correct
                }
                subject_results.append(result_entry)

                # 更新计数
                if is_correct:
                    subject_correct += 1
                    total_correct += 1
                total_samples += 1
                # time.sleep(1)  # 添加这行

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
            self.save_subject_logs(subject, subject_log_data, result_dir)
            time.sleep(30)  # 添加这行
        # 计算总体准确率
        overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0

        # 保存总体结果
        summary = {
            "model_indexes": model_choice,  # 模型索引列表
            "model_names": model_names,     # 模型名称列表
            "subjects": list(all_results.keys()),
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "details": {subj: {"accuracy": all_results[subj]["accuracy"],
                               "correct": all_results[subj]["correct"],
                               "total": all_results[subj]["total"]}
                        for subj in all_results}
        }

        summary_file = os.path.join(result_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印结果
        print("\n" + "=" * 50)
        print(f"CEval评估完成 - 总体准确率: {overall_acc:.2f}%")
        print("各科目准确率:")
        for subject, res in summary["details"].items():
            print(f"  {subject}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
        print("=" * 50)

        return overall_acc

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





    def extract_option(self, text):
        """
        使用正则表达式从生成的文本中提取选项（A、B、C或D）
        """
        # 将文本转换为大写以进行不区分大小写的匹配
        text_upper = text.upper()

        # 正则表达式模式列表（按优先级排序）
        patterns = [
            # 匹配括号中的选项，如 (A) 或 [B]
            r'[\(\[](A|B|C|D)[\)\]]',

            # 匹配冒号后的选项，如 "答案：A" 或 "正确答案: B"
            r'(?:答案|正确答案|正确选项|选项|选择|答案选项)[：:]\s*(A|B|C|D)',

            # 匹配"是"后的选项，如 "答案是A" 或 "正确答案是 B"
            r'(?:答案|正确答案|正确选项|选项|选择|答案选项)是\s*(A|B|C|D)',

            # 匹配单独的大写字母选项（确保前后没有其他字母）
            r'\b(A|B|C|D)\b',

            # 匹配任何位置的大写字母选项（最后的兜底选项）
            r'(A|B|C|D)'
        ]

        # 按优先级尝试匹配模式
        for pattern in patterns:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)

        # 如果所有方法都失败，返回随机选项
        return 'E'

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

    def save_csv(self, log_data, csv_path):
        with open(csv_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 添加MACS相关的三列
            writer.writerow([
                "question",
                "step", "model_name", "model_arch", "selected_word",
                "token", "prob", "logprob", "token_rank",
                "is_selected", "response_time_ms", "current_ans",
                "c_logit", "c_rank", "c_total"  # MACS分数
            ])

            question = log_data.get("question", "")
            for step in log_data['steps']:
                selected = step["selected_word"]
                for model in step["return_args"]:
                    # 获取MACS分数
                    c_logit = model.get("c_logit", 0.0)
                    c_rank = model.get("c_rank", 0.0)
                    c_total = model.get("c_total", 0.0)

                    for token_info in model["topk_token"]:
                        is_selected = int(token_info["token"] == selected)
                        writer.writerow([
                            question,
                            step["step"],
                            model["model_name"],
                            model["model_arch"],
                            selected,
                            token_info["token"],
                            round(token_info["prob"], 6),
                            round(token_info["logprob"], 6),
                            token_info["token_rank"],
                            is_selected,
                            model["response_time_ms"],
                            step["current_ans"],
                            round(c_logit, 6),  # logit投票支持度
                            round(c_rank, 6),  # Top-K命中情况
                            round(c_total, 6)  # 最终贡献度
                        ])

    def save_subject_logs(self, subject, log_data_list, result_dir):
        """保存整个学科的日志到JSON和CSV文件"""
        # 创建时间戳，确保同一学科的不同运行使用相同的时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON
        json_path = os.path.join(result_dir, f"{subject}_logs_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(log_data_list, f, indent=2, ensure_ascii=False)

        # 保存CSV
        csv_path = os.path.join(result_dir, f"{subject}_logs_{timestamp}.csv")
        with open(csv_path, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow([
                "question",
                "step", "model_name", "model_arch", "selected_word",
                "token", "prob", "logprob", "token_rank",
                "is_selected", "response_time_ms", "current_ans",
                "c_logit", "c_rank", "c_total"
            ])

            # 写入每个问题的数据
            for log_data in log_data_list:
                self.write_log_entry_to_csv(writer, log_data)

    def write_log_entry_to_csv(self, writer, log_data):
        """将单个问题的日志数据写入CSV"""
        question = log_data.get("question", "")
        for step in log_data['steps']:
            selected = step["selected_word"]
            for model in step["return_args"]:
                # 获取MACS分数
                c_logit = model.get("c_logit", 0.0)
                c_rank = model.get("c_rank", 0.0)
                c_total = model.get("c_total", 0.0)

                for token_info in model["topk_token"]:
                    is_selected = int(token_info["token"] == selected)
                    writer.writerow([
                        question,
                        step["step"],
                        model["model_name"],
                        model["model_arch"],
                        selected,
                        token_info["token"],
                        round(token_info["prob"], 6),
                        round(token_info["logprob"], 6),
                        token_info["token_rank"],
                        is_selected,
                        model["response_time_ms"],
                        step["current_ans"],
                        round(c_logit, 6),  # logit投票支持度
                        round(c_rank, 6),  # Top-K命中情况
                        round(c_total, 6)  # 最终贡献度
                    ])


# 使用示例
if __name__ == '__main__':
    # 实例化处理器
    handler = MultiModelHandler()


    # 选数据库中第i号模型     # 0：Qwen1.5-7B-Chat 2：Qwen2.5-7B-Instruct 3：GLM-4-9B-Chat 4：Meta-Llama-3.1-8B-Instruct
    model_choice_list =  [ [0, 2], [0, 3], [0, 4], [2, 3], [2, 4], [3, 4], [0, 2, 3], [0, 2, 4], [0, 3, 4], [2, 3,4],[0,2,3,4]]
    # model_choice_list =  [[0, 2, 3]]
    # model_choice_list =  [ [0, 2, 3], [0, 2, 4], [0, 3, 4], [2, 3,4],[0,2,3,4]]


    # model_choice_list =  [[0]]
    number_problems=20
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


        # CEval评估
        print("\n开始CEval评估...")
        # 选择要评估的科目（None表示全部科目）
        # subjects_to_evaluate = ["computer_network_val", "operating_system_val",
        #                         "computer_architecture_val",'high_school_biology_val',
        #                         'high_school_chemistry_val','high_school_chinese_val',
        #                         'high_school_geography_val','high_school_history_val',
        #                         'high_school_mathematics_val','high_school_physics_val',
        #                         'high_school_politics_val']
        subjects_to_evaluate = ['high_school_biology_val']
        # 每个科目评估的最大样本数（None表示全部）
        max_samples_per_subject = number_problems

        overall_acc = handler.evaluate_ceval(
            model_choice,
            args,
            subjects=subjects_to_evaluate,
            max_samples=max_samples_per_subject
        )
        print(f"CEval评估完成，总体准确率: {overall_acc:.2f}%")

    # nohup python