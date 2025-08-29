import os
import json
import re
import time
import csv
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from utils import  write_log_entry_to_csv, save_subject_logs
import torch
import gc

class BaseTester:
    """测试类的基类，封装公共功能"""

    def __init__(self, handler, dataset_path, results_dir,run_id=""):
        self.handler = handler
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.accuracy_history = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = os.path.join(self.results_dir, f"{self.timestamp}")
        # os.makedirs(self.result_dir, exist_ok=True)

        # 添加 run_id 作为目录名的一部分
        if run_id:
            self.result_dir = os.path.join(self.results_dir, f"{self.timestamp}_{run_id}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.result_dir = os.path.join(self.results_dir, f"{timestamp}")

        os.makedirs(self.result_dir, exist_ok=True)

    def _init_evaluation(self, model_choice):
        """初始化评估状态"""
        with open(self.handler.file_path, 'r', encoding='utf-8') as file:
            self.models_data = json.load(file)

        self.model_names = [self.models_data[i]["model_name"] for i in model_choice]
        self.accuracy_history = []
        self.active_model_choice = model_choice.copy()
        self.total_correct = 0
        self.total_samples = 0
        self.cumulative_correct = 0
        self.all_results = {}
        # ====== 新增：结果目录名加入 model_choice ======
        model_suffix = "_".join([str(i) for i in model_choice])
        new_result_dir = f"{self.result_dir}_{model_suffix}"
        if not os.path.exists(new_result_dir):
            os.rename(self.result_dir, new_result_dir)
        self.result_dir = new_result_dir
        # ==============================================
    def _handle_model_removal(self, removal_events):
        """处理模型剔除事件"""
        if removal_events:
            for event in removal_events:
                model_name = event["model_name"]
                model_index = next(
                    (idx for idx, model_idx in enumerate(self.active_model_choice)
                     if self.models_data[model_idx]["model_name"] == model_name
                     ), None)
                if model_index is not None:
                    self.active_model_choice.pop(model_index)

    def _update_accuracy(self, is_correct, subject, problem_id):
        """更新准确率统计"""
        if is_correct:
            self.total_correct += 1
            self.cumulative_correct += 1
        self.total_samples += 1

        current_accuracy = self.cumulative_correct / self.total_samples * 100
        self.accuracy_history.append({
            "subject": subject,
            "problem_id": problem_id,
            "accuracy": current_accuracy,
            "models": [self.models_data[idx]["model_name"] for idx in self.active_model_choice]
        })
        return current_accuracy

    def _record_problem_result(self, problem_record):
        """记录问题结果（如果启用了MACS管理）"""
        if hasattr(self.handler, 'macs_manager'):
            self.handler.macs_manager.all_problem_records.append(problem_record)

    def _save_subject_results(self, subject, subject_results, subject_log_data):
        """保存科目结果"""
        subject_acc = subject_results["correct"] / subject_results["total"] * 100
        self.all_results[subject] = {
            "accuracy": subject_acc,
            "correct": subject_results["correct"],
            "total": subject_results["total"],
            "results": subject_results["results"]
        }

        subject_file = os.path.join(self.result_dir, f"{subject}.json")
        with open(subject_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_results[subject], f, indent=2, ensure_ascii=False)

        save_subject_logs(subject, subject_log_data, self.result_dir)

    def _save_final_results(self):
        """保存最终结果和摘要"""
        overall_acc = self.total_correct / self.total_samples * 100 if self.total_samples > 0 else 0

        summary = {
            "model_names": self.model_names,
            "subjects": list(self.all_results.keys()),
            "overall_accuracy": overall_acc,
            "total_correct": self.total_correct,
            "total_samples": self.total_samples,
            "details": {subj: {
                "accuracy": self.all_results[subj]["accuracy"],
                "correct": self.all_results[subj]["correct"],
                "total": self.all_results[subj]["total"]
            } for subj in self.all_results},
            "final_models": [self.models_data[idx]["model_name"] for idx in self.active_model_choice]
        }

        if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
            summary["removal_events"] = self.handler.macs_manager.removal_events

        summary_file = os.path.join(self.result_dir, "summary.json")
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


        return overall_acc

    def _process_item(self, item, subject, problem_id, args, format_func, extract_func, get_answer_func):
        """处理单个问题项"""
        # 格式化问题
        question = format_func(item, subject)

        # 生成答案
        generated_answer, log_data, current_problem_macs = self.handler.generate_response(
            self.active_model_choice, question, args, problem_id=problem_id, subject=subject
        )

        # ===== 新增内存清理代码 =====
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # 强制垃圾回收
        gc.collect()
        # =========================

        # 检查剔除条件
        removal_events = []
        if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
            removal_events = self.handler.macs_manager.check_removal_condition(
                current_problem_macs, problem_id=problem_id, subject=subject
            )

        # 处理模型剔除
        self._handle_model_removal(removal_events)

        # 提取选项并检查正确性
        predicted_answer = extract_func(generated_answer)
        correct_answer = get_answer_func(item)
        is_correct = predicted_answer == correct_answer

        # 更新准确率
        current_accuracy = self._update_accuracy(is_correct, subject, problem_id)

        # 记录问题详细信息
        problem_record = {
            "subject": subject,
            "problem_id": problem_id,
            "is_correct": is_correct,
            "accuracy_so_far": current_accuracy,
            "removal_occurred": bool(removal_events)
        }

        if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
            problem_record["models"] = [
                {"name": name, "avg_macs": sum(scores) / len(scores), "scores": scores}
                for name, scores in current_problem_macs.items()
            ]

        if removal_events:
            problem_record["removal_events"] = removal_events

        self._record_problem_result(problem_record)


        return {
            "is_correct": is_correct,
            "log_data": log_data,
            "result_entry": {
                "id": problem_id,
                "question": item.get("question", ""),
                "correct_answer": correct_answer,
                "generated_text": generated_answer,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "current_models": [self.models_data[idx]["model_name"] for idx in self.active_model_choice]
            }
        }


class CEvalTester(BaseTester):
    """管理CEval测试过程"""

    def __init__(self, handler, ceval_val_path="./dataset/ceval-exam/val",
                 ceval_results_dir="./out/ceval_results", run_id=""):
        super().__init__(handler, ceval_val_path, ceval_results_dir, run_id="")
        self.ceval_val_path = ceval_val_path

    def format_ceval_question(self, item, subject):
        """格式化CEval问题为模型输入"""
        question = item["question"]
        options = {"A": item["A"], "B": item["B"], "C": item["C"], "D": item["D"]}
        options_str = "\n".join([f"{key}. {value}" for key, value in options.items()])
        subject_name = subject[:-4]
        return f"以下是一道{subject_name}的选择题，不输出其他任何内容，请直接输出答案选项（A、B、C或D）:\n\n{question}\n\n选项:\n{options_str}"
        # return f"以下是一道{subject_name}的选择题，你必须在回答的最后重申你的答案（A、B、C或D）:\n\n问题:{question}\n\n选项:\n{options_str}"

    import re

    def extract_option(self,text):
        """
        使用正则表达式从生成的文本中提取选项（A、B、C或D）
        优先从文本末尾匹配，适合模型在最后重申答案的场景
        """
        # 将文本转换为大写以进行不区分大小写的匹配
        text_upper = text.upper().strip()

        # 如果文本很短，直接尝试匹配整个文本
        if len(text_upper) <= 50:
            match = re.search(r'(A|B|C|D)$', text_upper)
            if match:
                return match.group(1)

        # 从文本末尾提取最后一部分（约50个字符）进行匹配
        # 这通常包含模型重申的答案
        end_text = text_upper[-50:] if len(text_upper) > 50 else text_upper

        # 正则表达式模式列表（按优先级排序）
        patterns = [
            # 匹配末尾的答案声明
            r'(?:答案|正确答案|正确选项|选项|选择|答案选项)[：:]\s*(A|B|C|D)\s*$',
            r'(?:答案|正确答案|正确选项|选项|选择|答案选项)是\s*(A|B|C|D)\s*$',

            # 匹配末尾的单独字母
            r'(A|B|C|D)\s*$',

            # 如果末尾没有找到，再尝试在整个文本中匹配
            r'(?:答案|正确答案|正确选项|选项|选择|答案选项)[：:]\s*(A|B|C|D)',
            r'(?:答案|正确答案|正确选项|选项|选择|答案选项)是\s*(A|B|C|D)',
            r'[\(\[](A|B|C|D)[\)\]]',
            r'\b(A|B|C|D)\b'
        ]

        # 先尝试在文本末尾匹配
        for pattern in patterns[:3]:  # 前三个模式专门用于末尾匹配
            match = re.search(pattern, end_text)
            if match:
                return match.group(1)

        # 如果末尾没有找到，再尝试在整个文本中匹配
        for pattern in patterns[3:]:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)

        # 如果所有方法都失败，返回None
        return None
    def evaluate(self, model_choice, args, subjects=None, max_samples=None):
        """在CEval验证集上评估模型性能"""
        self._init_evaluation(model_choice)

        if subjects is None:
            subjects = [d for d in os.listdir(self.dataset_path)
                        if os.path.isdir(os.path.join(self.dataset_path, d))]

        for subject in tqdm(subjects, desc="Subjects"):
            csv_path = os.path.join(self.dataset_path, subject)
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

            if max_samples > len(data):
                max_samples = len(data)

            subject_correct = 0
            subject_results = []
            subject_log_data = []

            for i, item in enumerate(tqdm(data, desc=f"处理 {subject}", leave=False)):
                if not self.active_model_choice:
                    print("所有模型已被剔除，终止评估")
                    break

                result = self._process_item(
                    item, subject, item["id"], args,
                    self.format_ceval_question,
                    lambda x: self.extract_option(x),
                    lambda x: x["answer"].upper()
                )

                if result["is_correct"]:
                    subject_correct += 1

                subject_results.append(result["result_entry"])
                subject_log_data.append(result["log_data"])

            subject_data = {
                "correct": subject_correct,
                "total": len(data),
                "results": subject_results
            }
            self._save_subject_results(subject, subject_data, subject_log_data)

            if not self.active_model_choice:
                break

        if hasattr(self.handler, 'macs_manager'):
            self.handler.macs_manager.save_removal_records(self.result_dir)

        return self._save_final_results()

    def predict_test_set(self, model_choice, args, subjects=None, max_samples=None):
        """
        在CEval测试集上生成预测结果
        返回格式: {学科: {问题ID: 预测答案, ...}, ...}
        """
        self._init_evaluation(model_choice)
        submission_results = {}
        detailed_results = {}

        if subjects is None:
            subjects = [d for d in os.listdir(self.dataset_path)
                        if os.path.isdir(os.path.join(self.dataset_path, d))]

        for subject in tqdm(subjects, desc="Subjects"):
            csv_path = os.path.join(self.dataset_path, subject)
            if not os.path.exists(csv_path):
                print(f"跳过 {subject} - CSV文件不存在: {csv_path}")
                continue

            data = []
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if all(field in row for field in ['id', 'question', 'A', 'B', 'C', 'D']):
                        data.append(row)

            if not data:
                print(f"跳过 {subject} - CSV文件中无有效数据")
                continue

            if max_samples and max_samples < len(data):
                data = data[:max_samples]

            subject_name = subject.replace(".csv", "")
            submission_results[subject_name] = {}
            detailed_results[subject_name] = {}

            for item in tqdm(data, desc=f"处理 {subject}", leave=False):
                if not self.active_model_choice:
                    print("所有模型已被剔除，终止预测")
                    break

                # 格式化问题
                question = self.format_ceval_question(item, subject)

                # 生成答案
                generated_answer, _, current_problem_macs = self.handler.generate_response(
                    self.active_model_choice, question, args,
                    problem_id=item["id"], subject=subject
                )

                # 内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()

                # 检查并处理模型剔除
                removal_events = []
                if hasattr(self.handler, 'macs_manager') and self.handler.macs_manager.enable_removal:
                    removal_events = self.handler.macs_manager.check_removal_condition(
                        current_problem_macs, problem_id=item["id"], subject=subject
                    )
                self._handle_model_removal(removal_events)

                # 提取预测答案
                predicted_answer = self.extract_option(generated_answer)

                # 记录结果
                problem_id = item["id"]
                submission_results[subject_name][problem_id] = predicted_answer

                detailed_results[subject_name][problem_id] = {
                    "question": item["question"],
                    "options": {"A": item["A"], "B": item["B"], "C": item["C"], "D": item["D"]},
                    "generated_text": generated_answer,
                    "predicted_answer": predicted_answer,
                    "models_used": [self.models_data[idx]["model_name"] for idx in self.active_model_choice],
                    "removal_occurred": bool(removal_events)
                }

                if removal_events:
                    detailed_results[subject_name][problem_id]["removal_events"] = removal_events

        # 保存结果
        submission_file = os.path.join(self.result_dir, "ceval_submission.json")
        with open(submission_file, 'w', encoding='utf-8') as f:
            json.dump(submission_results, f, indent=2, ensure_ascii=False)

        detailed_file = os.path.join(self.result_dir, "detailed_predictions.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=4, ensure_ascii=False)

        print(f"预测完成! 结果已保存至: {self.result_dir}")
        print(f"提交文件: {submission_file}")
        print(f"详细结果: {detailed_file}")

        return submission_results


class BoolQTester(BaseTester):
    """管理BoolQ测试过程"""

    def __init__(self, handler, boolq_path="./dataset/BoolQ",
                 boolq_results_dir="./out/boolq_results", run_id=""):
        super().__init__(handler, boolq_path, boolq_results_dir, run_id="")
        self.boolq_path = boolq_path

    def format_boolq_question(self, item, subject):
        """格式化BoolQ问题为模型输入"""
        title = item.get("title", "")
        passage = item["passage"]
        question = item["question"]

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
        text = text.lower().strip()
        if "true" in text or "yes" in text or "correct" in text:
            return "true"
        elif "false" in text or "no" in text or "incorrect" in text:
            return "false"
        else:
            match = re.search(r'\b(t|f|y|n)\b', text)
            if match:
                letter = match.group(1)
                if letter in ['t', 'y']:
                    return "true"
                elif letter in ['f', 'n']:
                    return "false"
            return None

    def evaluate(self, model_choice, args, max_samples=None):
        """在BoolQ数据集上评估模型性能"""
        self._init_evaluation(model_choice)
        data_file = os.path.join(self.dataset_path, "dev.jsonl")

        if not os.path.exists(data_file):
            print(f"BoolQ数据文件不存在: {data_file}")
            return 0.0

        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        if max_samples and max_samples < len(data):
            data = data[:max_samples]
        if max_samples > len(data):
            max_samples = len(data)
        subject = "boolq"
        subject_correct = 0
        subject_results = []
        subject_log_data = []

        for i, item in enumerate(tqdm(data, desc="处理BoolQ样本")):
            if not self.active_model_choice:
                print("所有模型已被剔除，终止评估")
                break

            result = self._process_item(
                item, subject, f"boolq_{i}", args,
                self.format_boolq_question,
                self.extract_bool_answer,
                lambda x: "true" if x["answer"] else "false"
            )

            if result["is_correct"]:
                subject_correct += 1

            subject_results.append(result["result_entry"])
            subject_log_data.append(result["log_data"])


        subject_data = {
            "correct": subject_correct,
            "total": len(data),
            "results": subject_results
        }
        self._save_subject_results(subject, subject_data, subject_log_data)

        if hasattr(self.handler, 'macs_manager'):
            self.handler.macs_manager.save_removal_records(self.result_dir)

        return self._save_final_results()


class SimpleMathTester(BaseTester):
    """管理SimpleMath测试过程"""

    def __init__(self, handler, simplemath_path="./dataset/grade_school_math",
                 simplemath_results_dir="./out/simplemath_results", run_id=""):
        super().__init__(handler, simplemath_path, simplemath_results_dir, run_id="")
        self.simplemath_path = simplemath_path

    def format_simplemath_question(self, item, subject):
        """格式化SimpleMath问题为模型输入"""
        question = item["question"]
        return f"请解答以下数学问题，并在最后一行以'#### '开头给出最终答案：\n\n问题: {question}\n解答过程:\n"

    def extract_math_answer(self, text):
        """从模型生成文本中提取数学答案"""
        lines = text.strip().split('\n')
        for line in reversed(lines):
            if line.startswith("#### "):
                answer_part = line[5:].strip()
                try:
                    return float(answer_part)
                except ValueError:
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer_part)
                    if numbers:
                        try:
                            return float(numbers[0])
                        except:
                            continue
        return None

    def get_true_answer(self, item):
        """从数据项中获取真实答案"""
        if "answer" in item:
            answer_lines = item["answer"].split("\n")
            for line in reversed(answer_lines):
                if line.startswith("#### "):
                    try:
                        return float(line[5:].strip())
                    except ValueError:
                        continue
        return None

    def evaluate(self, model_choice, args, max_samples=None):
        """在SimpleMath数据集上评估模型性能"""
        self._init_evaluation(model_choice)
        data_file = os.path.join(self.dataset_path, "train.jsonl")

        if not os.path.exists(data_file):
            print(f"SimpleMath数据文件不存在: {data_file}")
            return 0.0

        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        if max_samples and max_samples < len(data):
            data = data[:max_samples]
        if max_samples > len(data):
            max_samples = len(data)
        subject = "simplemath"
        subject_correct = 0
        subject_results = []
        subject_log_data = []

        for i, item in enumerate(tqdm(data, desc="处理SimpleMath样本")):
            if not self.active_model_choice:
                print("所有模型已被剔除，终止评估")
                break

            true_answer = self.get_true_answer(item)
            if true_answer is None:
                print(f"跳过问题 {i}，无法提取真实答案")
                continue

            result = self._process_item(
                item, subject, f"simplemath_{i}", args,
                self.format_simplemath_question,
                self.extract_math_answer,
                lambda x: true_answer
            )

            # 检查数值答案是否在容差范围内
            if result["result_entry"]["predicted_answer"] is not None:
                is_correct = abs(result["result_entry"]["predicted_answer"] - true_answer) < 1e-5
                result["is_correct"] = is_correct
                result["result_entry"]["is_correct"] = is_correct

            if result["is_correct"]:
                subject_correct += 1

            subject_results.append(result["result_entry"])
            subject_log_data.append(result["log_data"])


        subject_data = {
            "correct": subject_correct,
            "total": len(data),
            "results": subject_results
        }
        self._save_subject_results(subject, subject_data, subject_log_data)

        if hasattr(self.handler, 'macs_manager'):
            self.handler.macs_manager.save_removal_records(self.result_dir)

        return self._save_final_results()