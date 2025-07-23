import requests
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
import pdb
from PN import MultiModelHandler

def integrate_responses(single_ans):
    # 去除每个回答中的无用空格和<end>标记
    cleaned_answers = [ans.strip().replace('<end>', '')[:500] for ans in single_ans]

    # 使用"\n\n"将各个回答连接起来
    integrated_string = "\n\n".join(cleaned_answers)

    # 去除多余的空格和换行符
    integrated_string = integrated_string.replace('\n\n\n', '\n\n').strip()

    return integrated_string

class MultiModelHandlerThink:
    def __init__(self, num_model=None, eos_tokens=None, ports=None, max_workers=10):
        """
        初始化多模型处理器。

        :param num_model: 模型的数量
        :param eos_tokens: 终止标记的列表
        :param ports: 各模型的服务端口
        :param max_workers: 并发线程数
        """

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
            self.ports = [5050, 5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061, 5062]

        #初始化词级别模型
        self.handler = MultiModelHandler()
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def generate_response(self, model_choice, question, args):
        """
        生成最终回答。

        :param model_choice: 选择的模型编号列表
        :param question: 输入问题
        :param args: 参数配置
        :return: 最终生成的答案
        """
        texts = [None] * self.num_model
        for choice in model_choice:
            texts[choice] = self.call_template(question, self.ports[choice])

        # 特殊参数
        extra_args = {'topk': args['topk'], 'garbled': False}

        # 下一个单词
        new_word = ""
        # 回答长度
        len_of_tokens = 0
        # 总回答内容
        ans = ""
        
        # 为deepseek模型添加额外思考过程
        if 12 not in model_choice:
            return "ERROR"
        else:
            temp_model_choice = [x for x in model_choice if x != 12]

            #首先并行调用各大模型
            all_task = [
                    self.executor.submit(self.handler.generate_response, [choice], question, args)
                    for choice in temp_model_choice
            ]

            # 提取模型结果
            single_ans = []
            
            for future in as_completed(all_task):
                result = future.result()
                single_ans.append(result)
            print("single_ans:",single_ans)

            # 将各个模型的回答进行整合
            prompt = integrate_responses(single_ans)

            # 处理think模型
            choice = 12 
            texts[choice] = texts[choice] + prompt + "</think>\n\n"
            num = 0

            while new_word not in self.eos_tokens and num < 1000:
                result = self.call_app(texts[choice], self.ports[choice], choice, extra_args)
                print(result)
                new_word = max(result[1], key=lambda x: x[1])[0]
                texts[choice] += new_word
                ans = ans + new_word
                num = num + 1

        return ans
    
    def call_app(self, text, port, choice, extra_args):
        """
        调用单个模型的预测服务。

        :param text: 输入文本
        :param port: 服务端口
        :param choice: 模型编号
        :return: (choice, 模型预测结果)
        """
        
        url = f'http://10.154.22.10:{port}/predict'
        headers = {'Content-Type': 'application/json'}
        data = {'text': text, 'args': extra_args}

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return choice, result['prediction_values']
        else:
            print(f"Model_{choice}, Error: {response.status_code}")
            return choice, []

    def call_template(self, question, port):
        """
        调用单个模型的模板生成服务。

        :param question: 输入问题
        :param port: 服务端口
        :return: 模板文本
        """
        url = f'http://10.154.22.10:{port}/template'
        headers = {'Content-Type': 'application/json'}
        data = {'question': question}

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('text', '')
        else:
            print(f"Error: {response.status_code}")
            return ""

    def calculate_scores(self, data, args):
        """
        根据模型输出计算最终的选择。

        :param data: 模型输出数据
        :param args: 参数配置（包括模式选择）
        :return: （选定的词，得分）
        """
 
        scores = {}
        for _, values in data.items():
            for word, score in values:
                if word in self.eos_tokens:
                    word = "<end>"
                scores[word] = scores.get(word, 0) + score

        if not scores:
            return "<end>", 1.0

        max_score = max(scores.values())
        highest_scoring_words = [word for word, score in scores.items() if score == max_score]
        return random.choice(highest_scoring_words), max_score

# 使用示例
if __name__ == '__main__':
    # num_model = 20
    # eos_tokens = ['<|im_end|>', '<｜end▁of▁sentence｜>', '<|endoftext|>', '<|eot_id|>', '</s>', '<|user|>']
    ports = [5050, 5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061,5062]
    
    # 实例化处理器
    handler = MultiModelHandlerThink(ports=ports)
    
    # 输入参数示例
    # model_choice = [0,1,2,3,4,5,6,7,8,12]
    model_choice = [2,3,6,8,12]
    question = "关于核酸的叙述, 错误的是 :(A)细胞核中发生的转录过程有 RNA 聚合酶的参与,(B)植物细胞的线粒体和叶绿体中均可发生 DNA 的复制,(C)双链 DNA 分子中一条链上的磷酸和核糖是通过氢键连接的,(D)用甲基绿和吡罗红染色剂可观察 DNA 和 RNA 在细胞中的分布"
    args = {
        'max_len': 500,
        'topk': 3,
        'prefix': True,
        'soft': True,
        'log-info': True,
        'mode': 1
    }

    # 调用生成答案
    result = handler.generate_response(model_choice, question, args)
    print(f"Generated response: {result}")
