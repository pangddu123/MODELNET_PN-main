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
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from MACSManager import MACSManager
from main import MultiModelHandler


def run_parameter_test():
    """运行参数测试并收集结果"""
    # 定义要测试的参数范围
    param_grid = {
        'removal_threshold': [0.3, 0.4],
        'window_size': [3, 5],
        'consecutive_windows': [2, 3, 4],
        'use_relative_threshold': [True],
        'relative_threshold': [0.6, 0.65, 0.7],
    }

    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_combinations = list(itertools.product(*param_grid.values()))

    # 测试设置
    number_problems = 20
    number_subjects = 4
    model_choice = [0, 1, 2, 3]  # 固定模型组合

    # 结果存储
    results = []
    test_start_time = time.time()

    print(f"开始参数测试，共 {len(param_combinations)} 种组合")

    for i, params in enumerate(tqdm(param_combinations, desc="测试进度")):
        # 创建参数字典
        param_dict = dict(zip(param_names, params))

        # 创建处理程序
        handler = MultiModelHandler(
            enable_removal=True,
            **param_dict
        )

        # 固定参数
        args = {
            'max_len': 500,
            'top_k': 5,
            'prefix': False,
            'soft': True,
            'log-info': False,  # 关闭详细日志以加速测试
            'do_sample': True,
            'max_new_tokens': 10,
            'temperature': 0.8,
            'return_dict_in_generate': True,
            'output_scores': True,
            'top_p': None,
            'handel_next_token': True,
            'mode': 0
        }

        run_id = f"param_test_{i}"

        # 评估并收集结果
        combo_start_time = time.time()

        # 运行多个数据集的评估
        dataset_results = []
        for dataset_name, evaluate_func in [
            ("CEval", handler.evaluate_ceval),
            ("MMLU", handler.evaluate_mmlu),
            ("BoolQ", handler.evaluate_boolq),
        ]:
            # 重置MACS管理器状态
            handler.macs_manager = MACSManager(
                enable_removal=True,
                **param_dict
            )

            # 设置数据集特定参数
            if dataset_name == "CEval":
                folder_path = "./dataset/ceval-exam/val"
                subjects = [f for f in os.listdir(folder_path)
                            if f.endswith(".csv") and os.path.isfile(os.path.join(folder_path, f))]
                subjects = subjects[:number_subjects]
                accuracy = evaluate_func(
                    model_choice,
                    args,
                    subjects,
                    max_samples=number_problems,
                    run_id=run_id
                )
            elif dataset_name == "MMLU":
                folder_path = "./dataset/MMLU_ceval/data/val"
                subjects = [f for f in os.listdir(folder_path)
                            if f.endswith(".csv") and os.path.isfile(os.path.join(folder_path, f))]
                subjects = subjects[:number_subjects]
                accuracy = evaluate_func(
                    model_choice,
                    args,
                    subjects,
                    max_samples=number_problems,
                    run_id=run_id
                )
            else:  # BoolQ和SimpleMath
                accuracy = evaluate_func(
                    model_choice,
                    args,
                    max_samples=number_problems,
                    run_id=run_id
                )

            # 收集统计信息
            removal_events = handler.macs_manager.removal_events
            num_removals = len(removal_events)
            removal_types = [e['removal_type'] for e in removal_events]

            # 收集被移除的模型信息
            removed_models = [e['model_name'] for e in removal_events]

            # 分析模型移除频率
            model_removal_freq = {}
            model_tokens_before_removal = {}

            # 获取每个模型在被移除前生成的token数量
            for model_name in handler.macs_manager.model_macs_history:
                model_data = handler.macs_manager.model_macs_history[model_name]
                if model_data['removed']:
                    # 记录模型在被移除前生成的token数量
                    model_tokens_before_removal[model_name] = len(model_data['scores'])

            # 记录结果
            dataset_result = {
                **param_dict,
                'dataset': dataset_name,
                'accuracy': accuracy,
                'num_removals': num_removals,
                'macs_removals': removal_types.count('MACS'),
                'random_removals': removal_types.count('random'),
                'removed_models': ", ".join(removed_models),
                'model_removal_freq': json.dumps(model_removal_freq),
                'model_tokens_before_removal': json.dumps(model_tokens_before_removal)
            }

            dataset_results.append(dataset_result)

        # 合并所有数据集结果
        result = {
            **param_dict,
            'execution_time': time.time() - combo_start_time,
            'run_id': run_id,
            'dataset_results': json.dumps(dataset_results)
        }

        results.append(result)

        # 每5次测试保存一次结果
        if (i + 1) % 5 == 0:
            save_intermediate_results(results)

    # 保存最终结果
    save_final_results(results)
    print(f"参数测试完成，耗时: {time.time() - test_start_time:.2f}秒")
    return results


def save_intermediate_results(results):
    """保存中间结果"""
    df = pd.DataFrame(results)
    os.makedirs("./param_test_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"./param_test_results/intermediate_results_{timestamp}.csv", index=False)


def save_final_results(results):
    """保存最终结果并生成分析报告"""
    # 保存CSV
    df = pd.DataFrame(results)
    os.makedirs("./param_test_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"./param_test_results/final_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    # 生成分析报告
    generate_analysis_report(df, timestamp)

    return df


def generate_analysis_report(df, timestamp):
    """生成参数分析报告和可视化"""
    # 创建报告目录
    report_dir = f"./param_test_results/analysis_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)

    # 创建报告文件
    report_path = os.path.join(report_dir, f"analysis_report_{timestamp}.md")

    # 初始化报告内容
    report = f"# MACS参数测试分析报告\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"测试组合数: {len(df)}\n"
    report += f"平均执行时间: {df['execution_time'].mean():.2f}秒\n\n"

    # 提取所有数据集结果
    all_dataset_results = []
    for _, row in df.iterrows():
        dataset_results = json.loads(row['dataset_results'])
        for dataset_result in dataset_results:
            dataset_result['param_combination'] = row['run_id']
            all_dataset_results.append(dataset_result)

    dataset_df = pd.DataFrame(all_dataset_results)

    # 数据集总体统计
    report += "## 数据集总体统计\n"
    for dataset_name in ['CEval', 'MMLU', 'BoolQ', 'SimpleMath']:
        dataset_data = dataset_df[dataset_df['dataset'] == dataset_name]
        if not dataset_data.empty:
            report += f"\n### {dataset_name} 数据集\n"
            report += f"- 平均准确率: {dataset_data['accuracy'].mean():.4f}\n"
            report += f"- 平均剔除次数: {dataset_data['num_removals'].mean():.2f}\n"
            report += f"- 测试组合数: {len(dataset_data)}\n"

    # 参数重要性分析 - 准确率
    report += "\n## 参数对准确率的影响（按数据集）\n"

    for dataset_name in ['CEval', 'MMLU', 'BoolQ', 'SimpleMath']:
        dataset_data = dataset_df[dataset_df['dataset'] == dataset_name]
        if dataset_data.empty:
            continue

        report += f"\n### {dataset_name} 数据集\n"

        for param in ['removal_threshold', 'window_size', 'consecutive_windows',
                      'use_relative_threshold', 'relative_threshold']:
            if param in dataset_data.columns:
                grouped = dataset_data.groupby(param)['accuracy'].agg(['mean', 'std', 'count'])
                report += f"\n#### {param} 对准确率的影响\n"
                report += grouped.to_string() + "\n"

                # 创建准确率箱线图
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=dataset_data, x=param, y='accuracy')
                plt.title(f'{dataset_name}: {param} 对准确率的影响')
                plot_path = os.path.join(report_dir, f"{dataset_name}_accuracy_by_{param}_{timestamp}.png")
                plt.savefig(plot_path)
                plt.close()

                report += f"![]({os.path.basename(plot_path)})\n"

    # 模型移除分析
    report += "\n## 模型移除分析\n"

    # 分析每个模型被移除前的平均token数量
    model_token_stats = {}
    for _, row in dataset_df.iterrows():
        if row.get('model_tokens_before_removal'):
            try:
                token_data = json.loads(row['model_tokens_before_removal'])
                for model, tokens in token_data.items():
                    if model not in model_token_stats:
                        model_token_stats[model] = {'count': 0, 'total_tokens': 0}
                    model_token_stats[model]['count'] += 1
                    model_token_stats[model]['total_tokens'] += tokens
            except:
                pass

    report += "### 模型被移除前生成的token数量\n"
    if model_token_stats:
        report += "| 模型名称 | 平均token数 | 移除次数 |\n"
        report += "|----------|-------------|----------|\n"
        for model, stats in model_token_stats.items():
            avg_tokens = stats['total_tokens'] / stats['count']
            report += f"| {model} | {avg_tokens:.2f} | {stats['count']} |\n"
    else:
        report += "无模型被移除记录\n"

    # 模型token数量条形图
    if model_token_stats:
        plt.figure(figsize=(12, 6))
        models = []
        avg_tokens = []
        for model, stats in model_token_stats.items():
            models.append(model)
            avg_tokens.append(stats['total_tokens'] / stats['count'])

        plt.bar(models, avg_tokens)
        plt.title('模型被移除前平均生成的token数量')
        plt.xlabel('模型名称')
        plt.ylabel('平均token数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(report_dir, f"model_tokens_before_removal_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        report += f"\n![]({os.path.basename(plot_path)})\n"

    # 不同数据集上模型被移除前的token数量
    report += "\n### 不同数据集上模型被移除前的token数量\n"

    for dataset_name in ['CEval', 'MMLU', 'BoolQ', 'SimpleMath']:
        dataset_data = dataset_df[dataset_df['dataset'] == dataset_name]
        if dataset_data.empty:
            continue

        model_token_stats = {}
        for _, row in dataset_data.iterrows():
            if row.get('model_tokens_before_removal'):
                try:
                    token_data = json.loads(row['model_tokens_before_removal'])
                    for model, tokens in token_data.items():
                        if model not in model_token_stats:
                            model_token_stats[model] = {'count': 0, 'total_tokens': 0}
                        model_token_stats[model]['count'] += 1
                        model_token_stats[model]['total_tokens'] += tokens
                except:
                    pass

        if model_token_stats:
            report += f"\n#### {dataset_name} 数据集\n"
            report += "| 模型名称 | 平均token数 | 移除次数 |\n"
            report += "|----------|-------------|----------|\n"
            for model, stats in model_token_stats.items():
                avg_tokens = stats['total_tokens'] / stats['count']
                report += f"| {model} | {avg_tokens:.2f} | {stats['count']} |\n"

            # 创建图表
            plt.figure(figsize=(10, 6))
            models = list(model_token_stats.keys())
            avg_tokens = [stats['total_tokens'] / stats['count'] for stats in model_token_stats.values()]
            plt.bar(models, avg_tokens)
            plt.title(f'{dataset_name}: 模型被移除前平均生成的token数量')
            plt.xlabel('模型名称')
            plt.ylabel('平均token数量')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plot_path = os.path.join(report_dir, f"{dataset_name}_model_tokens_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            report += f"![]({os.path.basename(plot_path)})\n"

    # 参数交互分析 - 准确率热力图
    report += "\n## 参数交互分析\n"

    for dataset_name in ['CEval', 'MMLU', 'BoolQ', 'SimpleMath']:
        dataset_data = dataset_df[dataset_df['dataset'] == dataset_name]
        if dataset_data.empty:
            continue

        report += f"\n### {dataset_name} 数据集\n"

        # 准确率热力图 - 移除阈值 vs 窗口大小
        plt.figure(figsize=(12, 8))
        pivot_df = dataset_data.pivot_table(
            values='accuracy',
            index='removal_threshold',
            columns='window_size',
            aggfunc='mean'
        )
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title(f'{dataset_name}: 不同移除阈值和窗口大小组合的准确率')
        plot_path = os.path.join(report_dir, f"{dataset_name}_accuracy_heatmap_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        report += f"![]({os.path.basename(plot_path)})\n"

        # 准确率热力图 - 相对阈值 vs 窗口数
        if 'use_relative_threshold' in dataset_data.columns and dataset_data['use_relative_threshold'].any():
            plt.figure(figsize=(12, 8))
            relative_df = dataset_data[dataset_data['use_relative_threshold'] == True]
            pivot_df = relative_df.pivot_table(
                values='accuracy',
                index='relative_threshold',
                columns='consecutive_windows',
                aggfunc='mean'
            )
            sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu")
            plt.title(f'{dataset_name}: 不同相对阈值和连续窗口数组合的准确率')
            plot_path = os.path.join(report_dir, f"{dataset_name}_relative_heatmap_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            report += f"![]({os.path.basename(plot_path)})\n"

    # 保存报告
    with open(report_path, "w") as f:
        f.write(report)

    print(f"分析报告和图表已保存至: {report_dir}")
    return report_path


if __name__ == '__main__':
    # 运行参数测试
    test_results = run_parameter_test()

    # 打印最佳参数组合
    best_by_accuracy = test_results.sort_values('accuracy', ascending=False).iloc[0]
    best_by_removals = test_results.sort_values('num_removals', ascending=False).iloc[0]

    print("\n最佳准确率参数组合:")
    print(f"准确率: {best_by_accuracy['accuracy']:.4f}")
    print(f"参数: removal_threshold={best_by_accuracy['removal_threshold']}, "
          f"window_size={best_by_accuracy['window_size']}, "
          f"consecutive_windows={best_by_accuracy['consecutive_windows']}, "
          f"use_relative_threshold={best_by_accuracy['use_relative_threshold']}, "
          f"relative_threshold={best_by_accuracy['relative_threshold']}")

    print("\n最高剔除次数参数组合:")
    print(f"剔除次数: {best_by_removals['num_removals']}")
    print(f"参数: removal_threshold={best_by_removals['removal_threshold']}, "
          f"window_size={best_by_removals['window_size']}, "
          f"consecutive_windows={best_by_removals['consecutive_windows']}, "
          f"use_relative_threshold={best_by_removals['use_relative_threshold']}, "
          f"relative_threshold={best_by_removals['relative_threshold']}")