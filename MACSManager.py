import os
import math
import json
from datetime import datetime
import pandas as pd

class MACSManager:
    """管理MACS计算和模型剔除机制"""

    def __init__(self, enable_removal=False, removal_threshold=0.3,
                 window_size=5, consecutive_windows=3, max_removals=2):
        self.enable_removal = enable_removal
        self.removal_threshold = removal_threshold
        self.window_size = window_size
        self.consecutive_windows = consecutive_windows
        self.max_removals = max_removals

        # 状态跟踪
        self.removal_events = []
        self.all_problem_records = []
        self.model_macs_history = {}
        self.removal_count = 0
        self.current_models = []

    def initialize_models(self, model_names):
        """初始化模型状态"""
        self.current_models = model_names
        for model_name in model_names:
            if model_name not in self.model_macs_history:
                self.model_macs_history[model_name] = {
                    'scores': [],
                    'low_score_count': 0,
                    'removed': False
                }

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

        return c_total

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

    def save_removal_records(self, result_dir):
        """保存剔除记录和问题数据到文件"""
        if not self.enable_removal:
            return

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

