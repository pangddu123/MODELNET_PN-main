import os
import json
import re
import time
import csv
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from utils import extract_option, write_log_entry_to_csv, save_subject_logs

class CEvalTester:
    """管理CEval测试过程"""

    def __init__(self, handler, ceval_val_path="./dataset/ceval-exam/val",
                 ceval_results_dir="./out/ceval_results"):
        self.handler = handler
        self.ceval_val_path = ceval_val_path
        self.ceval_results_dir = ceval_results_dir
        os.makedirs(self.ceval_results_dir, exist_ok=True)

        # 测试状态
        self.accuracy_history = []

    def format_ceval_question(self, item, subject):
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

    def evaluate(self, model_choice, args, subjects=None, max_samples=None):
        """
        在CEval验证集上评估模型性能
        """
        with open(self.handler.file_path, 'r', encoding='utf-8') as file:
            models_data = json.load(file)

        model_names = [models_data[i]["model_name"] for i in model_choice]

        # 重置状态
        self.accuracy_history = []

        if subjects is None:
            subjects = [d for d in os.listdir(self.ceval_val_path)
                        if os.path.isdir(os.path.join(self.ceval_val_path, d))]

        all_results = {}
        total_correct = 0
        total_samples = 0
        cumulative_correct = 0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.ceval_results_dir, f"{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        # 初始化模型选择
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
                generated_answer, log_data, current_problem_macs = self.handler.generate_response(
                    active_model_choice, question, args, problem_id=item["id"], subject=subject
                )

                # 检查剔除条件（如果启用MACS管理）
                removal_events = []
                if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                    removal_events = self.handler.macs_manager.check_removal_condition(
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

                # 记录问题详细信息（如果启用MACS管理）
                problem_record = {
                    "subject": subject,
                    "problem_id": item["id"],
                    "is_correct": is_correct,
                    "accuracy_so_far": current_accuracy,
                    "removal_occurred": bool(removal_events)
                }

                if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                    problem_record["models"] = [
                        {
                            "name": name,
                            "avg_macs": sum(scores) / len(scores) if scores else 0,
                            "scores": scores
                        }
                        for name, scores in current_problem_macs.items()
                    ]

                if removal_events:
                    problem_record["removal_events"] = removal_events

                if hasattr(self.handler, 'macs_manager'):
                    self.handler.macs_manager.all_problem_records.append(problem_record)
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
            if hasattr(self.handler, 'macs_manager'):
                self.handler.macs_manager.save_removal_records(result_dir)

            if not active_model_choice:
                break


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
            "final_models": [models_data[idx]["model_name"] for idx in active_model_choice]
        }

        # 添加剔除事件信息（如果启用）
        if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
            summary["removal_events"] = self.handler.macs_manager.removal_events

        summary_file = os.path.join(result_dir, "summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 50)
        print(f"评估完成 - 总体准确率: {overall_acc:.2f}%")

        if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
            print(f"模型剔除事件: {len(self.handler.macs_manager.removal_events)}次")

        print("各科目准确率:")
        for subject, res in summary["details"].items():
            print(f"  {subject}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
        print("=" * 50)

        time.sleep(30)
        return overall_acc


class BoolQTester:
    """管理BoolQ测试过程"""

    def __init__(self, handler, boolq_path="./dataset/BoolQ",
                 boolq_results_dir="./out/boolq_results"):
        self.handler = handler
        self.boolq_path = boolq_path
        self.boolq_results_dir = boolq_results_dir
        os.makedirs(self.boolq_results_dir, exist_ok=True)

        # 测试状态
        self.accuracy_history = []

    def evaluate(self, model_choice, args, max_samples=None):
        """
        在BoolQ数据集上评估模型性能
        """
        # 加载数据集
        data_file = os.path.join(self.boolq_path, "dev.jsonl")
        if not os.path.exists(data_file):
            print(f"BoolQ数据文件不存在: {data_file}")
            return 0.0

        # 读取JSONL文件
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        if max_samples and max_samples < len(data):
            data = data[:max_samples]

        # 重置状态
        self.accuracy_history = []

        # 初始化模型选择
        active_model_choice = model_choice.copy()

        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.boolq_results_dir, f"boolq_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        total_correct = 0
        total_samples = 0
        cumulative_correct = 0
        results = []
        log_data = []

        for i, item in enumerate(tqdm(data, desc="处理BoolQ样本")):
            if not active_model_choice:
                print("所有模型已被剔除，终止评估")
                break

            # 格式化问题
            question = self.format_boolq_question(item)
            correct_answer = "true" if item["answer"] else "false"

            # 生成答案
            generated_answer, log_entry, current_problem_macs = self.handler.generate_response(
                active_model_choice, question, args, problem_id=f"boolq_{i}", subject="boolq"
            )

            # 检查剔除条件（如果启用MACS管理）
            removal_events = []
            if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                removal_events = self.handler.macs_manager.check_removal_condition(
                    current_problem_macs, problem_id=f"boolq_{i}", subject="boolq"
                )

            # 如果有模型被剔除，更新活动模型列表
            if removal_events:
                for event in removal_events:
                    model_name = event["model_name"]
                    # 找到被剔除模型的索引
                    with open(self.handler.file_path, 'r', encoding='utf-8') as file:
                        models_data = json.load(file)

                    model_index = next(
                        (idx for idx, model_idx in enumerate(active_model_choice)
                         if models_data[model_idx]["model_name"] == model_name),  # 括号闭合
                        None  # 默认值作为独立参数
                    )
                    if model_index is not None:
                        active_model_choice.pop(model_index)

                    # 提取和验证答案
                    predicted_answer = self.extract_bool_answer(generated_answer)
                    is_correct = predicted_answer == correct_answer if predicted_answer else False

                    # 更新准确率
                    if is_correct:
                        total_correct += 1
                    cumulative_correct += 1

                    total_samples += 1
                    current_accuracy = cumulative_correct / total_samples * 100
                    self.accuracy_history.append({
                        "problem_id": f"boolq_{i}",
                        "accuracy": current_accuracy,
                        "models": [models_data[idx]["model_name"] for idx in active_model_choice]
                    })

                    # 记录结果
                    result_entry = {
                        "id": f"boolq_{i}",
                        "title": item.get("title", ""),
                        "passage": item["passage"],
                        "question": item["question"],
                        "correct_answer": correct_answer,
                        "generated_text": generated_answer,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "current_models": [models_data[idx]["model_name"] for idx in active_model_choice]
                    }
                    results.append(result_entry)
                    log_data.append(log_entry)

                    # 记录问题详细信息（如果启用MACS管理）
                    problem_record = {
                        "subject": "boolq",
                        "problem_id": f"boolq_{i}",
                        "is_correct": is_correct,
                        "accuracy_so_far": current_accuracy,
                        "removal_occurred": bool(removal_events)
                    }

                    if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                        problem_record["models"] = [
                            {
                                "name": name,
                                "avg_macs": sum(scores) / len(scores) if scores else 0,
                                "scores": scores
                            }
                            for name, scores in current_problem_macs.items()
                        ]

                    if removal_events:
                        problem_record["removal_events"] = removal_events

                    if hasattr(self.handler, 'macs_manager'):
                        self.handler.macs_manager.all_problem_records.append(problem_record)

                    time.sleep(0.5)  # 避免请求过快

                    # 计算总体准确率
                    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0

                    # 保存结果
                    results_file = os.path.join(result_dir, "results.json")
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                    # 保存日志
                    save_subject_logs("boolq", log_data, result_dir)

                    # 保存剔除事件和问题记录
                    if hasattr(self.handler, 'macs_manager'):
                        self.handler.macs_manager.save_removal_records(result_dir)

                    # 保存准确率历史
                    accuracy_df = pd.DataFrame(self.accuracy_history)
                    accuracy_file = os.path.join(result_dir, "accuracy_history.xlsx")
                    accuracy_df.to_excel(accuracy_file, index=False)

                    # 保存摘要
                    summary = {
                        "total_samples": total_samples,
                        "total_correct": total_correct,
                        "overall_accuracy": overall_acc,
                        "removal_events": self.handler.macs_manager.removal_events if hasattr(self.handler,
                                                                                              'macs_manager') else [],
                        "final_models": [models_data[idx]["model_name"] for idx in active_model_choice]
                    }
                    summary_file = os.path.join(result_dir, "summary.json")
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)

                    print("\n" + "=" * 50)
                    print(f"BoolQ评估完成 - 总体准确率: {overall_acc:.2f}%")
                    print(f"处理样本数: {total_samples}")

                    if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                        print(f"模型剔除事件: {len(self.handler.macs_manager.removal_events)}次")

                    print("=" * 50)

        return overall_acc

    def format_boolq_question(self, item):
        """
        格式化BoolQ问题为模型输入
        """
        title = item.get("title", "")
        passage = item["passage"]
        question = item["question"]

        # 构建输入字符串
        input_str = f"根据以下文章回答问题：\n"
        if title:
            input_str += f"标题: {title}\n"
        input_str += f"文章: {passage}\n\n"
        input_str += f"问题: {question}\n"
        input_str += "答案只能是 'true' 或 'false'。\n\n"
        input_str += "答案: "

        return input_str

    def extract_bool_answer(self, text):
        """从模型生成文本中提取布尔答案"""
        # 尝试匹配各种可能的答案格式
        text = text.lower().strip()

        if "true" in text or "yes" in text or "correct" in text:
            return "true"
        elif "false" in text or "no" in text or "incorrect" in text:
            return "false"
        else:
            # 尝试匹配首字母
            match = re.search(r'\b(t|f|y|n)\b', text)
            if match:
                letter = match.group(1)
                if letter in ['t', 'y']:
                    return "true"
                elif letter in ['f', 'n']:
                    return "false"
            return None

class SimpleMathTester:
    """管理SimpleMath测试过程"""

    def __init__(self, handler, simplemath_path="./dataset/grade_school_math",
                 simplemath_results_dir="./out/simplemath_results"):
        self.handler = handler
        self.simplemath_path = simplemath_path
        self.simplemath_results_dir = simplemath_results_dir
        os.makedirs(self.simplemath_results_dir, exist_ok=True)

        # 测试状态
        self.accuracy_history = []

    def format_simplemath_question(self, item):
        """
        格式化SimpleMath问题为模型输入
        """
        question = item["question"]

        # 构建输入字符串
        input_str = f"请解答以下数学问题，并在最后一行以'#### '开头给出最终答案：\n\n"
        input_str += f"问题: {question}\n"
        input_str += "解答过程:\n"

        return input_str

    def extract_math_answer(self, text):
        """从模型生成文本中提取数学答案"""
        # 查找最后一个以"#### "开头的行
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if line.startswith("#### "):
                # 提取数字部分
                answer_part = line[5:].strip()
                # 尝试解析数字（可能是整数或小数）
                try:
                    # 先尝试直接转换为数字
                    return float(answer_part)
                except ValueError:
                    # 如果失败，尝试提取数字部分
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_part)
                    if numbers:
                        try:
                            return float(numbers[0])
                        except:
                            continue
        return None

    def evaluate(self, model_choice, args, max_samples=None):
        """
        在SimpleMath数据集上评估模型性能
        """
        # 加载数据集
        data_file = os.path.join(self.simplemath_path, "dev.jsonl")
        if not os.path.exists(data_file):
            print(f"SimpleMath数据文件不存在: {data_file}")
            return 0.0

        # 读取JSONL文件
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        if max_samples and max_samples < len(data):
            data = data[:max_samples]

        # 重置状态
        self.accuracy_history = []

        # 初始化模型选择
        active_model_choice = model_choice.copy()

        # 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join(self.simplemath_results_dir, f"simplemath_{timestamp}")
        os.makedirs(result_dir, exist_ok=True)

        total_correct = 0
        total_samples = 0
        cumulative_correct = 0
        results = []
        log_data = []

        for i, item in enumerate(tqdm(data, desc="处理SimpleMath样本")):
            if not active_model_choice:
                print("所有模型已被剔除，终止评估")
                break

            # 格式化问题
            question = self.format_simplemath_question(item)

            # 从答案中提取真实值（最后一个"####"后的数字）
            true_answer = None
            if "answer" in item:
                answer_lines = item["answer"].split("\n")
                for line in reversed(answer_lines):
                    if line.startswith("#### "):
                        try:
                            true_answer = float(line[5:].strip())
                            break
                        except ValueError:
                            continue

            # 如果无法提取真实答案，跳过该问题
            if true_answer is None:
                print(f"跳过问题 {i}，无法提取真实答案")
                continue

            # 生成答案
            generated_answer, log_entry, current_problem_macs = self.handler.generate_response(
                active_model_choice, question, args, problem_id=f"simplemath_{i}", subject="simplemath"
            )

            # 检查剔除条件（如果启用MACS管理）
            removal_events = []
            if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                removal_events = self.handler.macs_manager.check_removal_condition(
                    current_problem_macs, problem_id=f"simplemath_{i}", subject="simplemath"
                )

            # 如果有模型被剔除，更新活动模型列表
            if removal_events:
                for event in removal_events:
                    model_name = event["model_name"]
                    # 找到被剔除模型的索引
                    with open(self.handler.file_path, 'r', encoding='utf-8') as file:
                        models_data = json.load(file)
                    model_index = next(
                        (idx for idx, model_idx in enumerate(active_model_choice)
                         if models_data[model_idx]["model_name"] == model_name),  # 括号闭合
                        None  # 默认值作为独立参数
                    )
                    if model_index is not None:
                        active_model_choice.pop(model_index)

                    # 提取和验证答案
                    predicted_answer = self.extract_math_answer(generated_answer)
                    is_correct = False
                    if predicted_answer is not None:
                    # 允许数值计算中的微小误差
                        is_correct = abs(predicted_answer - true_answer) < 1e-5

                    # 更新准确率
                    if is_correct:
                        total_correct += 1
                    cumulative_correct += 1

                    total_samples += 1
                    current_accuracy = cumulative_correct / total_samples * 100
                    self.accuracy_history.append({
                        "problem_id": f"simplemath_{i}",
                        "accuracy": current_accuracy,
                        "models": [models_data[idx]["model_name"] for idx in active_model_choice]
                    })

                    # 记录结果
                    result_entry = {
                        "id": f"simplemath_{i}",
                        "question": item["question"],
                        "true_answer": true_answer,
                        "generated_text": generated_answer,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "current_models": [models_data[idx]["model_name"] for idx in active_model_choice]
                    }
                    results.append(result_entry)
                    log_data.append(log_entry)

                    # 记录问题详细信息（如果启用MACS管理）
                    problem_record = {
                        "subject": "simplemath",
                        "problem_id": f"simplemath_{i}",
                        "is_correct": is_correct,
                        "accuracy_so_far": current_accuracy,
                        "removal_occurred": bool(removal_events)
                    }

                    if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                        problem_record["models"] = [
                            {
                                "name": name,
                                "avg_macs": sum(scores) / len(scores) if scores else 0,
                                "scores": scores
                            }
                            for name, scores in current_problem_macs.items()
                        ]

                    if removal_events:
                        problem_record["removal_events"] = removal_events

                    if hasattr(self.handler, 'macs_manager'):
                        self.handler.macs_manager.all_problem_records.append(problem_record)

                    time.sleep(0.5)  # 避免请求过快

                    # 计算总体准确率
                    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0

                    # 保存结果
                    results_file = os.path.join(result_dir, "results.json")
                    with open(results_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                    # 保存日志
                    save_subject_logs("simplemath", log_data, result_dir)

                    # 保存剔除事件和问题记录
                    if hasattr(self.handler, 'macs_manager'):
                        self.handler.macs_manager.save_removal_records(result_dir)

                    # 保存准确率历史
                    accuracy_df = pd.DataFrame(self.accuracy_history)
                    accuracy_file = os.path.join(result_dir, "accuracy_history.xlsx")
                    accuracy_df.to_excel(accuracy_file, index=False)

                    # 保存摘要
                    summary = {
                        "total_samples": total_samples,
                        "total_correct": total_correct,
                        "overall_accuracy": overall_acc,
                        "removal_events": self.handler.macs_manager.removal_events if hasattr(self.handler,
                                                                                              'macs_manager') else [],
                        "final_models": [models_data[idx]["model_name"] for idx in active_model_choice]
                    }
                    summary_file = os.path.join(result_dir, "summary.json")
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)

                    print("\n" + "=" * 50)
                    print(f"SimpleMath评估完成 - 总体准确率: {overall_acc:.2f}%")
                    print(f"处理样本数: {total_samples}")

                    if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                        print(f"模型剔除事件: {len(self.handler.macs_manager.removal_events)}次")

                    print("=" * 50)

        return overall_acc