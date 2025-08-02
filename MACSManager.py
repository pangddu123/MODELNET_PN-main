import json
import os
import random
import math
from datetime import datetime

import pandas as pd


class MACSManager:
    """管理MACS计算和模型剔除机制"""

    def __init__(self, enable_removal=False, removal_threshold=0.25,
                 window_size=8, consecutive_windows=2, max_removals=3,
                 use_relative_threshold=True, relative_threshold=0.7,
                 random_removal_prob=0.0, random_removal_mode=False):  # 新增随机剔除参数
        """
        初始化MACS管理器

        :param enable_removal: 是否启用模型剔除机制
        :param removal_threshold: 绝对剔除阈值
        :param window_size: 滑动窗口大小
        :param consecutive_windows: 连续触发窗口数
        :param max_removals: 最大剔除模型数
        :param use_relative_threshold: 是否使用相对阈值
        :param relative_threshold: 相对阈值比例
        :param random_removal_prob: 随机剔除概率 (0.0-1.0)
        :param random_removal_mode: 是否启用纯随机剔除模式 (禁用MACS剔除)
        """
        self.enable_removal = enable_removal
        self.removal_threshold = removal_threshold
        self.window_size = window_size
        self.consecutive_windows = consecutive_windows
        self.max_removals = max_removals
        self.use_relative_threshold = use_relative_threshold
        self.relative_threshold = relative_threshold
        self.random_removal_prob = random_removal_prob  # 随机剔除概率
        self.random_removal_mode = random_removal_mode  # 纯随机剔除模式

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
                # 使用对数概率差计算c_logit
                logprob_diff = found_token['logprob'] - max_logprob
                # 将概率差转换为0-1范围的分数
                if logprob_diff >= 0:
                    c_logit = 1.0  # 目标token是最佳token
                else:
                    # 使用sigmoid转换: 范围(-∞,0] -> (0,0.5]
                    c_logit = 1 / (1 + math.exp(-logprob_diff))
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
        alpha = 0.7  # logit分量的权重 (提高重要性)
        beta = 1 - alpha  # rank分量的权重
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

        # 纯随机剔除模式
        if self.random_removal_mode:
            return self.perform_random_removal(problem_id, subject)

        # MACS剔除模式 (可能包含随机剔除)
        best_model_score = -1

        # 首先计算本轮最佳模型分数（仅当使用相对阈值时）
        if self.use_relative_threshold:
            for scores in current_problem_macs.values():
                if scores:
                    avg = sum(scores) / len(scores)
                    if avg > best_model_score:
                        best_model_score = avg

        for model_name, macs_scores in current_problem_macs.items():
            if self.model_macs_history[model_name]['removed']:
                continue

            # 计算当前问题平均分
            current_avg = sum(macs_scores) / len(macs_scores) if macs_scores else 0
            self.model_macs_history[model_name]['scores'].append(current_avg)

            # 维护窗口大小
            if len(self.model_macs_history[model_name]['scores']) > self.window_size:
                self.model_macs_history[model_name]['scores'].pop(0)

            # 检查剔除条件
            if len(self.model_macs_history[model_name]['scores']) >= self.consecutive_windows:
                last_n = self.model_macs_history[model_name]['scores'][-self.consecutive_windows:]
                avg_score = sum(last_n) / len(last_n)

                # 绝对阈值条件
                absolute_low = avg_score < self.removal_threshold

                # 相对阈值条件（仅在启用时检查）
                relative_low = False
                if self.use_relative_threshold and best_model_score > 0:
                    relative_low = avg_score < best_model_score * self.relative_threshold

                # 决定是否剔除
                if absolute_low and (not self.use_relative_threshold or relative_low):
                    # 随机剔除检查：有一定概率跳过MACS剔除
                    if self.random_removal_prob > 0 and random.random() > self.random_removal_prob:
                        continue

                    models_to_remove.append(model_name)
                    self.model_macs_history[model_name]['removed'] = True
                    self.removal_count += 1

                    # 记录剔除原因
                    reason = "绝对阈值" if not self.use_relative_threshold else "绝对+相对阈值"
                    threshold_info = f"{self.removal_threshold}" if not self.use_relative_threshold else f"{self.removal_threshold} & {best_model_score * self.relative_threshold:.3f}"

                    print(f"🚨 模型 {model_name} 被剔除 | "
                          f"窗口平均分: {avg_score:.3f} < 阈值: {threshold_info} "
                          f"(条件: {reason})")

        # 随机剔除（在MACS模式中作为补充）
        if self.random_removal_prob > 0 and not self.random_removal_mode:
            random_removals = self.perform_random_removal(problem_id, subject)
            if random_removals:
                models_to_remove.extend([r['model_name'] for r in random_removals])

        removal_events = []
        for model_name in models_to_remove:
            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "subject": subject,
                "problem_id": problem_id,
                "model_name": model_name,
                "window_scores": self.model_macs_history[model_name]['scores'][
                                 -self.consecutive_windows:] if not self.random_removal_mode else [],
                "threshold": self.removal_threshold,
                "relative_threshold_used": self.use_relative_threshold,
                "relative_threshold_value": self.relative_threshold if self.use_relative_threshold else None,
                "best_model_score": best_model_score if self.use_relative_threshold else None,
                "remaining_models": [m for m in self.current_models if m != model_name],
                "removal_type": "random" if self.random_removal_mode or (
                            self.random_removal_prob > 0 and model_name in [r['model_name'] for r in
                                                                            removal_events]) else "MACS"
            }
            self.removal_events.append(event)
            removal_events.append(event)

        return removal_events

    def perform_random_removal(self, problem_id, subject):
        """执行随机剔除"""
        models_to_remove = []
        removal_events = []

        # 获取尚未被剔除的模型
        available_models = [model for model in self.current_models
                            if not self.model_macs_history[model]['removed']]

        if not available_models:
            return removal_events

        # 计算实际剔除概率，考虑最大剔除数量限制
        effective_prob = min(self.random_removal_prob,
                             (self.max_removals - self.removal_count) / len(available_models))

        for model_name in available_models:
            # 检查是否达到最大剔除数量
            if self.removal_count >= self.max_removals:
                break

            # 根据概率决定是否剔除
            if random.random() < effective_prob:
                self.model_macs_history[model_name]['removed'] = True
                self.removal_count += 1
                models_to_remove.append(model_name)

                print(f"🎲 模型 {model_name} 被随机剔除 | "
                      f"概率: {effective_prob:.2f}")

        for model_name in models_to_remove:
            event = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "subject": subject,
                "problem_id": problem_id,
                "model_name": model_name,
                "window_scores": [],
                "threshold": self.random_removal_prob,
                "relative_threshold_used": False,
                "relative_threshold_value": None,
                "best_model_score": None,
                "remaining_models": [m for m in self.current_models if m != model_name],
                "removal_type": "random"
            }
            self.removal_events.append(event)
            removal_events.append(event)

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
                models = row['models']

                if isinstance(models, list):
                    for model_info in models:
                        model_data.append({
                            "subject": row['subject'],
                            "problem_id": row['problem_id'],
                            "model_name": model_info.get('name', ''),
                            "avg_macs": model_info.get('avg_macs', 0),
                            "scores": json.dumps(model_info.get('scores', [])),
                            "is_correct": row['is_correct'],
                            "accuracy_so_far": row['accuracy_so_far'],
                            "removal_occurred": row['removal_occurred'],
                            "removal_type": row.get('removal_type', 'none')
                        })
                elif isinstance(models, dict):
                    for model_name, model_info in models.items():
                        model_data.append({
                            "subject": row['subject'],
                            "problem_id": row['problem_id'],
                            "model_name": model_name,
                            "avg_macs": model_info.get('avg_macs', 0),
                            "scores": json.dumps(model_info.get('scores', [])),
                            "is_correct": row['is_correct'],
                            "accuracy_so_far": row['accuracy_so_far'],
                            "removal_occurred": row['removal_occurred'],
                            "removal_type": row.get('removal_type', 'none')
                        })
                else:
                    print(f"⚠️ 未知的models类型: {type(models)}")
                    continue

            model_df = pd.DataFrame(model_data)
            excel_file = os.path.join(result_dir, "all_problems_macs.xlsx")
            model_df.to_excel(excel_file, index=False)

            print(f"💾 保存剔除记录 | 事件: {len(self.removal_events)} 问题记录: {len(model_data)}")