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

from main import MultiModelHandler


# 假设所有必要的导入和类定义已经存在
# 这里只展示参数测试部分
def run_parameter_test():
    """运行参数测试并收集结果"""
    # 定义要测试的参数范围
    param_grid = {
        'removal_threshold': [0.3,0.35,  0.4],
        'window_size': [3, 5, 8],
        'consecutive_windows': [2, 3, 4],
        'use_relative_threshold': [True, False],
        'relative_threshold': [0.6, 0.65, 0.7],
        # 'random_removal_prob': [0.0, 0.05, 0.1],
        # 'random_removal_mode': [False]  # 固定为False，因为True是纯随机模式
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

        # 只测试CEval数据集（可以扩展其他数据集）
        ceval_folder_path = "./dataset/ceval-exam/val"
        ceval_subjects_to_evaluate = [f for f in os.listdir(ceval_folder_path)
                                      if f.endswith(".csv") and os.path.isfile(os.path.join(ceval_folder_path, f))]
        ceval_subjects_to_evaluate = ceval_subjects_to_evaluate[:number_subjects]

        # 运行评估
        try:
            accuracy = handler.evaluate_ceval(
                model_choice,
                args,
                ceval_subjects_to_evaluate,
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
            for model_name in set(removed_models):
                model_removal_freq[model_name] = removed_models.count(model_name)

            # 记录结果
            result = {
                **param_dict,
                'accuracy': accuracy,
                'num_removals': num_removals,
                'macs_removals': removal_types.count('MACS'),
                'random_removals': removal_types.count('random'),
                'execution_time': time.time() - combo_start_time,
                'run_id': run_id,
                'removed_models': ", ".join(removed_models),  # 保存被移除的模型列表
                'model_removal_freq': json.dumps(model_removal_freq)  # 保存移除频率统计
            }

            results.append(result)

            # 每5次测试保存一次结果
            if (i + 1) % 5 == 0:
                save_intermediate_results(results)

        except Exception as e:
            print(f"参数组合 {i} 测试失败: {str(e)}")
            result = {
                **param_dict,
                'accuracy': 0,
                'num_removals': 0,
                'macs_removals': 0,
                'random_removals': 0,
                'execution_time': 0,
                'run_id': run_id,
                'removed_models': "",
                'model_removal_freq': "{}",
                'error': str(e)
            }
            results.append(result)

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
    report += f"平均准确率: {df['accuracy'].mean():.4f}\n"
    report += f"平均剔除次数: {df['num_removals'].mean():.2f}\n"
    report += f"平均执行时间: {df['execution_time'].mean():.2f}秒\n\n"

    # 参数重要性分析 - 准确率
    report += "## 参数对准确率的影响\n"

    # 分组分析每个参数对准确率的影响
    for param in ['removal_threshold', 'window_size', 'consecutive_windows',
                  'use_relative_threshold', 'relative_threshold', 'random_removal_prob']:
        if param in df.columns:
            grouped = df.groupby(param)['accuracy'].agg(['mean', 'std', 'count'])
            report += f"\n### {param} 对准确率的影响\n"
            report += grouped.to_string() + "\n"

            # 创建准确率箱线图
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=param, y='accuracy')
            plt.title(f'{param} 对准确率的影响')
            plot_path = os.path.join(report_dir, f"accuracy_by_{param}_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()

            report += f"![{param}对准确率的影响]({os.path.basename(plot_path)})\n"

    # 参数重要性分析 - 剔除次数
    report += "\n## 参数对剔除次数的影响\n"

    for param in ['removal_threshold', 'window_size', 'consecutive_windows',
                  'use_relative_threshold', 'relative_threshold', 'random_removal_prob']:
        if param in df.columns:
            grouped = df.groupby(param)['num_removals'].agg(['mean', 'std', 'count'])
            report += f"\n### {param} 对剔除次数的影响\n"
            report += grouped.to_string() + "\n"

            # 创建剔除次数箱线图
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x=param, y='num_removals')
            plt.title(f'{param} 对剔除次数的影响')
            plot_path = os.path.join(report_dir, f"removals_by_{param}_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()

            report += f"![{param}对剔除次数的影响]({os.path.basename(plot_path)})\n"

    # 模型移除分析
    report += "\n## 模型移除分析\n"

    # 分析每个模型被移除的频率
    model_freq = {}
    for _, row in df.iterrows():
        if row.get('model_removal_freq'):
            try:
                freq_dict = json.loads(row['model_removal_freq'])
                for model, count in freq_dict.items():
                    model_freq[model] = model_freq.get(model, 0) + count
            except:
                pass

    # 按频率排序
    sorted_freq = sorted(model_freq.items(), key=lambda x: x[1], reverse=True) if model_freq else []

    report += "### 模型被移除频率统计\n"
    if sorted_freq:
        report += "| 模型名称 | 被移除次数 |\n"
        report += "|----------|------------|\n"
        for model, count in sorted_freq:
            report += f"| {model} | {count} |\n"
    else:
        report += "无模型被移除记录\n"

    # 模型移除频率条形图
    if sorted_freq:
        plt.figure(figsize=(10, 6))
        model_names = [item[0] for item in sorted_freq]
        frequencies = [item[1] for item in sorted_freq]
        plt.bar(model_names, frequencies)
        plt.title('模型被移除频率')
        plt.xlabel('模型名称')
        plt.ylabel('被移除次数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_path = os.path.join(report_dir, f"model_removal_freq_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        report += f"\n![模型被移除频率]({os.path.basename(plot_path)})\n"

    # 分析不同参数下模型的移除情况
    report += "\n### 不同参数对模型移除的影响\n"

    if model_freq:
        # 对于每个模型，分析在不同参数下的移除频率
        for model in model_freq.keys():
            report += f"\n#### 模型 {model} 的移除分析\n"

            # 分析不同移除阈值下的移除频率
            report += "不同移除阈值下的移除频率:\n"
            threshold_data = []
            for threshold in sorted(df['removal_threshold'].unique()):
                # 计算该模型在特定阈值下的移除频率
                freq = 0
                for _, row in df[df['removal_threshold'] == threshold].iterrows():
                    if row.get('model_removal_freq'):
                        try:
                            freq_dict = json.loads(row['model_removal_freq'])
                            freq += freq_dict.get(model, 0)
                        except:
                            pass
                threshold_data.append((threshold, freq))

            # 添加到报告
            report += "| 移除阈值 | 移除次数 |\n"
            report += "|----------|----------|\n"
            for threshold, freq in threshold_data:
                report += f"| {threshold} | {freq} |\n"

            # 创建折线图
            plt.figure(figsize=(10, 6))
            thresholds = [item[0] for item in threshold_data]
            freqs = [item[1] for item in threshold_data]
            plt.plot(thresholds, freqs, marker='o')
            plt.title(f'移除阈值对模型 {model} 移除次数的影响')
            plt.xlabel('移除阈值')
            plt.ylabel('移除次数')
            plt.grid(True)
            plot_path = os.path.join(report_dir, f"removal_by_threshold_{model}_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()
            report += f"![移除阈值对模型{model}的影响]({os.path.basename(plot_path)})\n"

    # 参数交互分析 - 准确率热力图
    report += "\n## 参数交互分析\n"

    # 准确率热力图 - 移除阈值 vs 窗口大小
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot_table(
        values='accuracy',
        index='removal_threshold',
        columns='window_size',
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title('不同移除阈值和窗口大小组合的准确率')
    plot_path = os.path.join(report_dir, f"accuracy_heatmap_threshold_window_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    report += f"\n### 移除阈值和窗口大小对准确率的影响\n"
    report += f"![准确率热力图]({os.path.basename(plot_path)})\n"

    # 准确率热力图 - 相对阈值 vs 窗口数
    if 'use_relative_threshold' in df.columns and df['use_relative_threshold'].any():
        plt.figure(figsize=(12, 8))
        relative_df = df[df['use_relative_threshold'] == True]
        pivot_df = relative_df.pivot_table(
            values='accuracy',
            index='relative_threshold',
            columns='consecutive_windows',
            aggfunc='mean'
        )
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="YlGnBu")
        plt.title('不同相对阈值和连续窗口数组合的准确率')
        plot_path = os.path.join(report_dir, f"accuracy_heatmap_relative_window_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        report += f"\n### 相对阈值和连续窗口数对准确率的影响\n"
        report += f"![相对阈值热力图]({os.path.basename(plot_path)})\n"

    # 剔除次数散点图
    report += "\n### 剔除阈值与剔除次数的关系\n"
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='removal_threshold',
        y='num_removals',
        hue='use_relative_threshold',
        size='relative_threshold',
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title('剔除阈值与剔除次数的关系')
    plt.grid(True)
    plot_path = os.path.join(report_dir, f"removals_scatter_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    report += f"![剔除阈值与剔除次数的关系]({os.path.basename(plot_path)})\n"

    # 执行时间分析
    report += "\n## 执行时间分析\n"

    # 窗口设置对执行时间的影响
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x='consecutive_windows',
        y='execution_time',
        hue='window_size'
    )
    plt.title('不同窗口设置下的执行时间')
    plot_path = os.path.join(report_dir, f"execution_time_boxplot_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    report += f"![不同窗口设置下的执行时间]({os.path.basename(plot_path)})\n"

    # 执行时间与准确率的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='execution_time',
        y='accuracy',
        hue='num_removals',
        size='removal_threshold',
        sizes=(20, 200),
        alpha=0.7
    )
    plt.title('执行时间与准确率的关系')
    plt.grid(True)
    plot_path = os.path.join(report_dir, f"time_vs_accuracy_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    report += f"\n### 执行时间与准确率的关系\n"
    report += f"![执行时间与准确率的关系]({os.path.basename(plot_path)})\n"

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
          f"relative_threshold={best_by_accuracy['relative_threshold']}, "
          f"random_removal_prob={best_by_accuracy['random_removal_prob']}")

    print("\n最高剔除次数参数组合:")
    print(f"剔除次数: {best_by_removals['num_removals']}")
    print(f"参数: removal_threshold={best_by_removals['removal_threshold']}, "
          f"window_size={best_by_removals['window_size']}, "
          f"consecutive_windows={best_by_removals['consecutive_windows']}, "
          f"use_relative_threshold={best_by_removals['use_relative_threshold']}, "
          f"relative_threshold={best_by_removals['relative_threshold']}, "
          f"random_removal_prob={best_by_removals['random_removal_prob']}")