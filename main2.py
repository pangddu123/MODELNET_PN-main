import atexit
import logging
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MultiModelHandler")


class MultiModelHandler:
    """多模型协作处理类（完整实现）"""

    def __init__(self, num_model=None, eos_tokens=None, ports=None, max_workers=10,
                 enable_removal=False, removal_threshold=0.3, window_size=5,
                 consecutive_windows=3, max_removals=2):
        # 文件路径配置
        self.file_path = "./model_info.json"
        self.save_dir = "./out/saved_logs"
        os.makedirs(self.save_dir, exist_ok=True)

        # 状态保存文件
        self.state_file = "./evaluation_state.json"
        self.current_state = self.load_state()

        # 注册退出时的状态保存
        atexit.register(self.save_state)

        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 初始化MACS管理器
        self.macs_manager = MACSManager(
            enable_removal=enable_removal,
            removal_threshold=removal_threshold,
            window_size=window_size,
            consecutive_windows=consecutive_windows,
            max_removals=max_removals
        )

        # 初始化测试器
        self.ceval_tester = CEvalTester(self)
        self.boolq_tester = BoolQTester(self)
        self.mmlu_tester = CEvalTester(
            self,
            ceval_val_path="./dataset/MMLU_ceval/data/val",
            ceval_results_dir="./out/mmlu_results"
        )
        self.simpleMath_tester = SimpleMathTester(self)

        # 生成过程监控参数
        self.max_repetition_count = 5  # 最大允许重复次数
        self.repetition_window = 3  # 重复检测窗口大小
        self.max_meaningless_tokens = 10  # 最大无意义token数量

    def load_state(self):
        """加载保存的状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载状态失败: {str(e)}")
        return {
            "current_model_choice": None,
            "current_subject": None,
            "completed_problems": [],
            "start_time": datetime.now().isoformat()
        }

    def save_state(self):
        """保存当前状态"""
        try:
            self.current_state["last_save"] = datetime.now().isoformat()
            with open(self.state_file, 'w') as f:
                json.dump(self.current_state, f, indent=2)
            logger.info("状态已保存")
        except Exception as e:
            logger.error(f"保存状态失败: {str(e)}")

    def update_state(self, model_choice, subject, problem_id):
        """更新状态"""
        self.current_state["current_model_choice"] = model_choice
        self.current_state["current_subject"] = subject
        if problem_id not in self.current_state["completed_problems"]:
            self.current_state["completed_problems"].append(problem_id)
        self.save_state()

    def call_app_with_retry(self, text, info, extra_args, max_retries=3):
        """带重试机制的API调用"""
        url = f"{info['model_url']}/predict"
        headers = {'Content-Type': 'application/json'}
        data = {'text': text, 'args': extra_args}
        model_eos_token = info.get('EOS', '')

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(url, json=data, headers=headers, timeout=30)
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000

                if response.status_code == 200:
                    # 检查响应内容有效性
                    response_data = response.json()
                    if self.is_valid_response(response_data, info['model_arch']):
                        # 处理非Transformers架构的模型
                        if info['model_arch'] != "transformers":
                            result = {'prediction_values': [], 'args': {}, 'sample_result': []}

                            # 处理主要响应
                            first_response = response_data['response'][0]
                            token = first_response['token']
                            logprob = first_response['logprob']
                            prob = math.exp(logprob)

                            # 处理空token情况
                            if first_response.get('bytes') == []:
                                token = model_eos_token

                            result['sample_result'].append([token, prob, logprob])

                            # 处理top_logprobs
                            for item in response_data['response'][0].get('top_logprobs', []):
                                token = item['token']
                                logprob = item['logprob']
                                prob = math.exp(logprob)

                                # 处理空token情况
                                if item.get('bytes') == []:
                                    token = model_eos_token

                                result['prediction_values'].append([token, prob, logprob])

                            return info["model_name"], result, elapsed_ms, model_eos_token
                        else:
                            # Transformers架构直接返回
                            return info["model_name"], response_data, elapsed_ms, model_eos_token
                    else:
                        logger.warning(f"模型 {info['model_name']} 返回无效响应")
                        raise ValueError("Invalid model response")
                else:
                    # 服务器错误，等待后重试
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"模型 {info['model_name']} 请求失败 (状态码 {response.status_code})，"
                                   f"{max_retries - attempt - 1}次重试剩余，等待 {wait_time:.1f}秒")
                    time.sleep(wait_time)
            except (requests.exceptions.RequestException, ConnectionError) as e:
                # 网络错误，等待后重试
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"模型 {info['model_name']} 连接错误: {str(e)}，"
                               f"{max_retries - attempt - 1}次重试剩余，等待 {wait_time:.1f}秒")
                time.sleep(wait_time)
            except ValueError as e:
                # 内容无效错误，直接重试
                wait_time = (2 ** attempt) * 0.5
                logger.warning(f"模型 {info['model_name']} 内容无效: {str(e)}，"
                               f"{max_retries - attempt - 1}次重试剩余，等待 {wait_time:.1f}秒")
                time.sleep(wait_time)
            except Exception as e:
                # 其他错误，直接返回空结果
                logger.error(f"模型 {info['model_name']} 未知错误: {str(e)}")
                return info["model_name"], {'sample_result': []}, 0, model_eos_token

        # 重试全部失败
        logger.error(f"模型 {info['model_name']} 请求失败，返回空结果")
        return info["model_name"], {'sample_result': []}, 0, model_eos_token

    def is_valid_response(self, response_data, model_arch):
        """验证模型响应是否有效"""
        # 检查基本结构
        if not response_data:
            return False

        if model_arch != "transformers":
            required_keys = ['response']
            if not all(key in response_data for key in required_keys):
                return False

            # 检查响应内容
            if not response_data['response']:
                return False

            first_response = response_data['response'][0]
            if 'token' not in first_response or 'logprob' not in first_response:
                return False

            # 检查top_logprobs
            if 'top_logprobs' in first_response:
                for item in first_response['top_logprobs']:
                    if 'token' not in item or 'logprob' not in item:
                        return False
        else:
            # Transformers模型检查
            required_keys = ['sample_result']
            if not all(key in response_data for key in required_keys):
                return False

            if not response_data['sample_result']:
                return False

        return True

    def call_template(self, question, info):
        """调用模板API"""
        url = f"{info['model_url']}/template"
        headers = {'Content-Type': 'application/json'}
        data = {'question': question}

        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json().get('text', '')
            else:
                logger.warning(f"模板API错误: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"模板API调用失败: {str(e)}")
            return ""

    def generate_response(self, model_choice, question, args, problem_id=None, subject=None):
        """生成响应（带错误处理和重复检测）"""
        # 验证参数
        val, info = utils.validate_args(args)
        if not val:
            return info, {}, {}

        # 加载模型信息
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # 更新当前模型列表
        model_names = [data[i]["model_name"] for i in model_choice]
        self.models_info = [data[i] for i in model_choice]
        self.eos_tokens = [info['EOS'] for info in self.models_info]

        # 应用模板
        texts = [self.call_template(question, info) for info in self.models_info]

        # 初始化生成状态
        recent_tokens = []  # 跟踪最近生成的token
        repetition_count = 0  # 当前重复计数
        meaningless_token_count = 0  # 连续无意义token计数
        new_word = ""
        len_of_tokens = 0
        ans = ""

        # 错误恢复策略
        recovery_attempts = 0
        max_recovery_attempts = 2
        original_temp = args.get('temperature', 0.8)  # 保存原始温度

        # 存储当前问题中各模型的MACS分数
        current_problem_macs = {model_name: [] for model_name in
                                model_names} if self.macs_manager.enable_removal else {}

        log_data = {
            "question": question,
            "model_names": model_names,
            "steps": []
        }

        # 主生成循环
        while True:
            # 退出条件检查
            exit_conditions = [
                new_word in ['<end>', '</s>', '<|endoftext|>'],  # EOS tokens
                len_of_tokens > args['max_len'],
                repetition_count >= self.max_repetition_count,
                meaningless_token_count >= self.max_meaningless_tokens
            ]

            if any(exit_conditions):
                logger.warning(f"生成终止: EOS={exit_conditions[0]}, "
                               f"长度={exit_conditions[1]}, "
                               f"重复={exit_conditions[2]}, "
                               f"无意义={exit_conditions[3]}")
                break

            len_of_tokens += 1
            next_words = {}
            return_args = []

            # 提交任务（使用带重试的API调用）
            all_task = [
                self.executor.submit(self.call_app_with_retry, texts[i], info, args)
                for i, info in enumerate(self.models_info)
            ]

            # 收集结果
            for future in as_completed(all_task):
                model_name, model_output, response_time, eos_token = future.result()
                info = next((m for m in self.models_info if m['model_name'] == model_name), {})
                model_arch = info.get("model_arch", "")

                # 处理模型输出
                if args['return_dict_in_generate'] is not True:
                    topk = sorted(model_output.get('sample_result', []), key=lambda x: x[1], reverse=True)[
                        :args['top_k']]
                elif args.get('handel_next_token', False):
                    topk = model_output.get('prediction_values', [])
                    topk = sorted(topk, key=lambda x: x[1], reverse=True)[:args['top_k']]
                else:
                    topk = sorted(model_output.get('sample_result', []), key=lambda x: x[1], reverse=True)[
                        :args['top_k']]

                # 格式化topk结果
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

                if args.get('mode', 0) == 1:
                    next_words[model_name] = [[word[0], 1] for word in topk]
                else:
                    next_words[model_name] = [[word[0], word[1]] for word in topk]

            # 检查模型响应质量
            valid_responses = 0
            for model_data in return_args:
                if model_data['topk_token']:
                    valid_responses += 1

            # 如果大多数模型返回无效响应，尝试恢复
            if valid_responses < len(self.models_info) // 2:
                logger.warning(f"模型响应质量低: {valid_responses}/{len(self.models_info)} 有效响应")

                if recovery_attempts < max_recovery_attempts:
                    recovery_attempts += 1
                    logger.info(f"尝试恢复策略 #{recovery_attempts}")

                    # 策略1: 增加温度以增加多样性
                    args['temperature'] = min(1.5, args.get('temperature', 0.8) + 0.3)
                    logger.info(f"提高温度至 {args['temperature']}")

                    # 策略2: 回退部分生成文本
                    if len(texts[0]) > 20:
                        rollback_length = min(10, len(texts[0]) // 3)
                        for i in range(len(texts)):
                            texts[i] = texts[i][:-rollback_length]
                        ans = ans[:-rollback_length]
                        logger.info(f"回退 {rollback_length} 个字符")

                    continue
                else:
                    logger.error("恢复尝试失败，终止生成")
                    break

            # 重置恢复尝试计数器
            recovery_attempts = 0
            args['temperature'] = original_temp  # 恢复原始温度

            # 计算新词
            new_word, score = self.calculate_scores(next_words, args)

            # 检测和处理重复模式
            recent_tokens.append(new_word)
            if len(recent_tokens) > self.repetition_window:
                recent_tokens.pop(0)

            # 检查重复模式
            if len(recent_tokens) == self.repetition_window:
                if all(token == new_word for token in recent_tokens):
                    repetition_count += 1
                    logger.warning(f"检测到重复模式: '{new_word}' (计数: {repetition_count})")

                    # 重复处理策略: 强制选择非重复token
                    non_repeat_candidates = []
                    for model_data in return_args:
                        for token_data in model_data['topk_token']:
                            if token_data['token'] != new_word:
                                non_repeat_candidates.append(token_data)

                    if non_repeat_candidates:
                        # 选择最高概率的非重复token
                        non_repeat_candidates.sort(key=lambda x: x['prob'], reverse=True)
                        new_word = non_repeat_candidates[0]['token']
                        logger.info(f"强制选择非重复token: '{new_word}'")
                else:
                    repetition_count = max(0, repetition_count - 1)

            # 检测无意义token (空格、标点等)
            if new_word.strip() in ['', '.', ',', ';', ':', '!', '?', ' ']:
                meaningless_token_count += 1
            else:
                meaningless_token_count = max(0, meaningless_token_count - 1)

            # 更新生成文本
            for i in range(len(texts)):
                texts[i] += new_word
            ans += new_word

            # 计算每个模型的MACS贡献分数（如果启用）
            if self.macs_manager.enable_removal:
                for model_data in return_args:
                    c_total = self.macs_manager.calculate_macs(model_data, new_word, args['top_k'])
                    model_name = model_data['model_name']
                    if model_name in current_problem_macs:  # 确保模型在列表中
                        current_problem_macs[model_name].append(c_total)

            # 记录日志
            log_data['steps'].append({
                "step": len_of_tokens,
                "next_words": next_words,
                "selected_word": new_word,
                "score": score,
                "return_args": return_args,
                "current_texts": texts.copy(),
                "current_ans": ans
            })

            # 日志输出
            if args.get('log-info', False):
                print(f"[Step {len_of_tokens}] new_word: {new_word} | next_words: {next_words}")

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
        """评估CEval数据集（带断点续跑）"""
        # 实现类似evaluate_mmlu的逻辑
        pass

    def evaluate_mmlu(self, model_choice, args, subjects=None, max_samples=None):
        """增强容错能力的MMLU评估"""
        # 加载当前状态
        state = self.load_state()

        # 如果状态中存在未完成的评估，继续执行
        if (state["current_model_choice"] == model_choice and
                state["current_subject"] and
                state["completed_problems"]):
            logger.info(f"恢复评估: 模型组合 {model_choice}, 科目 {state['current_subject']}")
            logger.info("已完成问题: %s", len(state['completed_problems']))


            # 继续之前的评估
            return self._evaluate_subject(
                model_choice,
                state["current_subject"],
                args,
                max_samples,
                state["completed_problems"]
            )

        # 否则开始新的评估
        MMLU_folder_path = "./dataset/MMLU_ceval/data/val"
        MMLU_subjects_to_evaluate = [f for f in os.listdir(MMLU_folder_path)
                                     if f.endswith(".csv") and os.path.isfile(os.path.join(MMLU_folder_path, f))]
        if subjects:
            MMLU_subjects_to_evaluate = [s for s in MMLU_subjects_to_evaluate if s in subjects]

        # 限制评估科目数量
        if max_samples and isinstance(max_samples, int):
            MMLU_subjects_to_evaluate = MMLU_subjects_to_evaluate[:max_samples]

        # 随机打乱科目顺序
        random.shuffle(MMLU_subjects_to_evaluate)

        overall_results = {}
        for subject in tqdm(MMLU_subjects_to_evaluate, desc="评估科目"):
            subject_name = subject.replace(".csv", "")
            logger.info(f"开始评估科目: {subject_name}")

            # 重置状态
            self.current_state = {
                "current_model_choice": model_choice,
                "current_subject": subject_name,
                "completed_problems": [],
                "start_time": datetime.now().isoformat()
            }
            self.save_state()

            # 评估当前科目
            subject_acc = self._evaluate_subject(
                model_choice,
                subject_name,
                args,
                max_samples
            )

            overall_results[subject_name] = subject_acc

            # 完成科目评估后重置状态
            self.current_state = {
                "current_model_choice": None,
                "current_subject": None,
                "completed_problems": [],
                "last_completed_subject": subject_name
            }
            self.save_state()

        # 计算整体准确率
        total_acc = sum(overall_results.values()) / len(overall_results) if overall_results else 0
        logger.info(f"整体准确率: {total_acc:.4f}")
        return total_acc

    def _evaluate_subject(self, model_choice, subject, args, max_samples, completed_problems=None):
        """评估单个科目（带断点续跑）"""
        completed_problems = completed_problems or []
        csv_path = os.path.join(self.mmlu_tester.ceval_val_path, f"{subject}.csv")

        if not os.path.exists(csv_path):
            logger.error(f"科目文件不存在: {csv_path}")
            return 0

        try:
            df = pd.read_csv(csv_path)
            if max_samples is not None and len(df) > max_samples:
                df = df.head(max_samples)
        except Exception as e:
            logger.error(f"读取CSV失败: {str(e)}")
            return 0

        correct_count = 0
        total_count = 0

        # 创建进度条，跳过已完成的问题
        pbar = tqdm(total=len(df), desc=f"评估 {subject}")
        pbar.update(len(completed_problems))

        for idx, row in df.iterrows():
            # 跳过已完成的问题
            if idx in completed_problems:
                continue

            try:
                # 更新状态（标记当前正在处理的问题）
                self.update_state(model_choice, subject, idx)

                # 评估当前问题
                question = row['question']
                options = [row['A'], row['B'], row['C'], row['D']]
                correct_answer = row['answer']

                # 生成响应
                response, log_data, _ = self.generate_response(
                    model_choice,
                    question,
                    args,
                    problem_id=idx,
                    subject=subject
                )

                # 提取选项
                selected_option = utils.extract_option(response)

                # 验证答案
                if selected_option == correct_answer:
                    correct_count += 1
                total_count += 1

                # 记录日志
                utils.save_subject_logs(subject, log_data, self.save_dir)

                # 更新进度
                pbar.update(1)
                time.sleep(0.5)  # 添加延迟避免服务器过载

            except Exception as e:
                logger.error(f"问题 {idx} 评估失败: {str(e)}")
                # 保存当前状态后退出
                self.save_state()
                logger.info("已保存当前状态，程序退出")
                return correct_count / total_count if total_count > 0 else 0

        # 完成评估
        accuracy = correct_count / len(df) if len(df) > 0 else 0
        logger.info(f"科目 {subject} 评估完成，准确率: {accuracy:.4f}")
        return accuracy

    def evaluate_boolq(self, model_choice, args, max_samples=1000):
        """评估BoolQ数据集"""
        # 实现类似evaluate_mmlu的逻辑
        pass

    def evaluate_simpleMath(self, model_choice, args, max_samples=1000):
        """评估简单数学问题"""
        # 实现类似evaluate_mmlu的逻辑
        pass


# 示例使用
if __name__ == '__main__':
    # 实例化处理器
    handler = MultiModelHandler(
        enable_removal=False,
        removal_threshold=0.3,
        window_size=5,
        consecutive_windows=3,
        max_removals=3
    )

    # 模型组合列表
    model_choice_list = [[1]]  # 简化示例

    number_problems = 20
    number_subjects = 5

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
            'output_scores': True,
            'top_p': None,
            'handel_next_token': True,
            'mode': 0
        }

        # 执行评估（自动恢复中断的评估）
        overall_acc = handler.evaluate_mmlu(model_choice, args, max_samples=number_problems)