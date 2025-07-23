import requests
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
from PN import MultiModelHandler
import pdb
import re

def extract_model_number(text):
    # 正则表达式用于匹配 "模型" 后可以跟零个或多个空格和数字
    pattern = r'模型\s*(\d+)'

    # 使用re.findall查找所有匹配项
    matches = re.findall(pattern, text)

    # 返回匹配结果，作为整数列表
    return [int(match) for match in matches]

def generate_prompt(question, single_ans):
    """
    生成用于评估多个答案的提示词。

    参数:
        question: 原始问题。
        single_ans: 一个包含多个模型回答的列表。

    返回:
        str: 生成的提示词。
    """
    # 检查单个答案列表是否为空
    if not single_ans:
        return "没有可评估的答案。"

    # 构建回答部分
    answers_section = "\n".join(f"***模型 {i + 1}***: {answer.strip()}\n" for i, answer in enumerate(single_ans))

    # 生成提示词
    prompt = f"""
        原问题：{question.strip()}

        以下是几个模型对该问题的回答：
        {answers_section}
        **由于以上模型能力不足，它们的回答很可能存在错误。**

        请您根据原问题与这些模型的回答，最后出您的答案。

        **注意：请确保您按照原问题的回答要求作答！**
    """

    return prompt

class SMultiModelHandler:
    def __init__(self, num_model=None, eos_tokens=None, ports=None, max_workers=10):
        """
        初始化多模型处理器。

        :param num_model: 模型的数量
        :param eos_tokens: 终止标记的列表
        :param ports: 各模型的服务端口
        :param max_workers: 并发线程数

        """
        # 实例化处理器

        if num_model is not None:
            self.num_model = num_model
        else:
            self.num_model = 20
        
        if eos_tokens is not None:
            self.eos_tokens = eos_tokens
        else:
            self.eos_tokens = ['<|im_end|>', '<｜end▁of▁sentence｜>', '<|endoftext|>', '<|eot_id|>', '</s>', '<|user|>']
        
        if ports is not None:
            self.ports = ports
        else:
            self.ports = [5050, 5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061]
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        #初始化词级别模型
        self.handler = MultiModelHandler()

    def s_generate_response(self, model_choice, question, args):
        """
        生成最终回答。

        :param model_choice: 选择的模型编号列表
        :param question: 输入问题
        :param args: 参数配置
        :return: 最终生成的答案
        """

        #首先并行调用各大模型
        all_task = [
                self.executor.submit(self.handler.generate_response, [choice], question, args)
                for choice in model_choice
        ]

        # 提取模型结果
        single_ans = []
        
        for future in as_completed(all_task):
            result = future.result()
            single_ans.append(result)

        pdb.set_trace()
        print("single_ans:",single_ans)
        # 对结果进行比较
        prompt = generate_prompt(question,single_ans)
        result = self.handler.generate_response(model_choice, prompt, args)

        return result
        # print(single_ans)
        # print(f"模型评价结果：{result}")
        # try:
        #     # 提取模型编号
        #     model_number = extract_model_number(result)
        #     print(f"正则匹配结果：{model_number[-1]}")
        #     return single_ans[model_number[-1] - 1]
        # except: 
        #     #若出现异常则随机选择一个答案
        #     print(f"匹配异常")
        #     return random.choice(single_ans)
    
# 使用示例
if __name__ == '__main__':
    
    # 实例化处理器
    shandler = SMultiModelHandler()
    
    # 输入参数示例
    model_choice = [2, 3, 6, 8]
    question = "What is the capital of France?"
    args = {
        'max_len': 500,
        'topk': 3,
        'prefix': False,
        'soft': False,
        'log-info': False,
        'mode': 0
    }

    # 调用生成答案
    result = shandler.s_generate_response(model_choice, question, args)
    print(result)
