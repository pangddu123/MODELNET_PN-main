import csv
import os
import re

import requests
import json
import numpy as np
from datetime import datetime

    
def softmax_probability(data):
    # 提取权重部分
    weights = np.array([item[1] for item in data])

    # 计算 softmax
    exp_weights = np.exp(weights - np.max(weights))  # 减去最大值以避免溢出
    probabilities = exp_weights / np.sum(exp_weights)

    # 将结果与对应的标签组合
    results = [[item[0], prob] for item, prob in zip(data, probabilities)]

    return results

def filter_prefixes(data):
    # 提取字符和权重
    words = [item[0] for item in data]
    prefix_words = set()  # 用于存储前缀字符
    non_prefix_words = set()  # 用于存储非前缀但独立的字符

    # 遍历每个字符，检查它是否是其他字符的前缀
    for word in words:
        is_prefix = any(other.startswith(word) for other in words if other != word)
        if is_prefix:
            prefix_words.add(word)  # 如果是前缀，则添加到前缀集合
        else:
            # 检查该字符是否是任何其他字符的前缀
            is_prefix_of_any = any(word.startswith(other) for other in words if other != word)
            if not is_prefix_of_any:
                non_prefix_words.add(word)  # 如果不是任何字符的前缀，则添加到独立字符集合

    # 生成结果，保留前缀字符和独立字符的信息
    filtered_data = [item for item in data if item[0] in prefix_words or item[0] in non_prefix_words]

    return filtered_data


def validate_args(args):
    """
    验证参数设置是否规范
    
    参数:
        args (dict): 参数字典
        
    返回:
        tuple: (bool, str) 第一个元素表示是否验证通过，第二个元素是错误信息（如果有）
    """
    # 定义参数规范
    rules = {
        'max_len': {'type': int, 'min': 1, 'max': 10000},
        'max_new_tokens': {'type': int, 'min': 1, 'max': 10000},
        'top_k': {'type': (int, type(None)), 'min': 0, 'max': 100},
        'top_p': {'type': (int, float, type(None)), 'min': 0, 'max': 100},
        'temperature': {'type': (float, int), 'min': 0, 'max': 1},
        'prefix': {'type': bool},
        'soft': {'type': bool},
        'log-info': {'type': bool},
        'do_sample': {'type': bool},
        'return_dict_in_generate': {'type': bool},
        'output_scores': {'type': bool},
        'mode': {'type': int, 'allowed': [0, 1]}
    }
    
    # 检查必填参数
    required_params = ['max_len', 'top_k', 'prefix', 'soft', 'log-info', 'do_sample']
    for param in required_params:
        if param not in args:
            return False, f"缺少必需参数: {param}"
    
    # 检查参数类型和取值范围
    for param, value in args.items():
        if param not in rules:
            continue  # 跳过未定义规则的参数
            
        rule = rules[param]
        
        # 检查类型
        expected_types = rule['type'] if isinstance(rule['type'], tuple) else (rule['type'],)
        if not isinstance(value, expected_types):
            return False, f"参数 {param} 类型错误，期望 {expected_types}，实际 {type(value)}"
        
        # 检查数值范围
        if isinstance(value, (int, float)) and 'min' in rule and 'max' in rule:
            if not (rule['min'] <= value <= rule['max']):
                return False, f"参数 {param} 值 {value} 超出允许范围 [{rule['min']}, {rule['max']}]"
        
        # 检查允许值
        if 'allowed' in rule and value not in rule['allowed']:
            return False, f"参数 {param} 值 {value} 不在允许值 {rule['allowed']} 中"
    
    # 检查互斥参数
    if args.get('top_p') is not None and args.get('top_k') == 0:
        return False, "top_p 和 top_k=0 不能同时设置"
    
    if args.get('do_sample') is False and args.get('temperature') != 1:
        return False, "当 do_sample=False 时，temperature 必须为 1"
    
    return True, "参数验证通过"





def save_subject_logs( subject, log_data_list, result_dir):
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
            write_log_entry_to_csv(writer, log_data)

def write_log_entry_to_csv(writer, log_data):
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
                    round(c_total, 6) ])
