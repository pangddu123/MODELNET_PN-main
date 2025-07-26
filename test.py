import os
import json
import time
import csv
from datetime import datetime
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
        result_dir = os.path.join(self.ceval_results_dir, f"ceval_{timestamp}")
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
        print(f"CEval评估完成 - 总体准确率: {overall_acc:.2f}%")

        if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
            print(f"模型剔除事件: {len(self.handler.macs_manager.removal_events)}次")

        print("各科目准确率:")
        for subject, res in summary["details"].items():
            print(f"  {subject}: {res['accuracy']:.2f}% ({res['correct']}/{res['total']})")
        print("=" * 50)

        time.sleep(30)
        return overall_acc
