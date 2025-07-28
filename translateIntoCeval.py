import os
import pandas as pd
from pathlib import Path


def convert_mmlu_to_ceval(input_base: str, output_base: str) -> None:
    """
    将MMLU格式的CSV数据集转换为Ceval格式

    参数:
    input_base: MMLU数据集的根目录路径
    output_base: 转换后Ceval格式的输出根目录

    特点:
    1. 保留原始目录结构
    2. 生成唯一ID: {父目录名}_{文件名}_{行号}
    3. 自动处理所有子目录中的CSV文件
    """
    # 确保输出目录存在
    Path(output_base).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(input_base):
        for file in files:
            if file.endswith(".csv"):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_base)
                output_dir = os.path.join(output_base, rel_path)
                output_path = os.path.join(output_dir, file)

                # 创建输出目录
                os.makedirs(output_dir, exist_ok=True)

                # 读取无表头CSV[1,7](@ref)
                df = pd.read_csv(input_path, header=None,
                                 names=['question', 'A', 'B', 'C', 'D', 'answer'])

                # 生成唯一ID[6](@ref)
                parent_dir = os.path.basename(root)
                filename_stem = Path(file).stem
                df['id'] = [f"{i}" for i in range(len(df))]

                # 调整列顺序为Ceval格式[5](@ref)
                ceval_df = df[['id', 'question', 'A', 'B', 'C', 'D', 'answer']]

                # 保存转换结果
                ceval_df.to_csv(output_path, index=False)
                print(f"Converted {input_path} -> {output_path}")


# 使用示例
if __name__ == "__main__":
    convert_mmlu_to_ceval(
        input_base="./dataset/MMLU/data",
        output_base="./dataset/MMLU_ceval/data"
    )