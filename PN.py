import requests
import math
import random
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import utils
from openai import OpenAI
import pdb

class MultiModelHandler:
    def __init__(self, num_model=None, eos_tokens=None, ports=None, max_workers=1):
        """
        初始化多模型处理器。

        :param model_info: 模型的信息
        :param max_workers: 并发线程数
        """
        # 读取模型配置文件(***后续修正为数据库连接***)
        self.file_path = "./model_info.json"
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def generate_response(self, model_choice, question, extra_args):
        """
        生成最终回答。

        :param model_choice: 选择的模型编号列表
        :param question: 输入问题
        :param args: 参数配置
        :return: 最终生成的答案
        """

        # 验证参数信息是否正确
        val, info = utils.validate_args(args)
        if not val:
            return info
        
        # 通过数据库获取模型信息（需根据数据库实现进行修正）
        # pdb.set_trace()
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        self.models_info =[data[i] for i in model_choice]
        self.eos_tokens = [info['EOS'] for info in self.models_info]

        # 初始化推理字符
        texts = [None] * len(self.models_info)
        for i, choice in enumerate(model_choice):
            # 读取利用对应路由进行编码
            texts[i] = self.call_template(question, self.models_info[i])

        # 下一个单词
        new_word = ""
        # 回答长度
        len_of_tokens = 0
        # 总回答内容
        ans = ""
                
        while new_word not in ['<end>']:
            if len_of_tokens > args['max_len'] or any(token in new_word for token in self.eos_tokens if token):
                break


            len_of_tokens += 1
            next_words = {}
        
            # 并发调用模型
            all_task = [
                self.executor.submit(self.call_app, texts[i], info, extra_args)
                for i,info in enumerate(self.models_info)
            ]

            return_args = []

            # 提取模型结果
            for future in as_completed(all_task):

                result = future.result()
                print(result)
                if args['return_dict_in_generate'] is not True:
                    topk_token = sorted(result[1]['response']['sample_result'], key=lambda x: x[1], reverse=True)[:args['top_k']]
                elif args['handel_next_token']:
                    topk_token = sorted(result[1]['response']['prediction_values'], key=lambda x: x[1], reverse=True)[:args['top_k']]

                else:
                    topk_token = sorted(result[1]['response']['sample_result'], key=lambda x: x[1], reverse=True)[:args['top_k']]
                
                return_args.append([result[0],result[1]['response']['args']])
                # 若为投票则权重结果选设置为1
                if args['mode'] == 1:           
                    topk_token = [[word, 1] for word, _ in topk_token]

                # 是否进行前缀修正
                if args.get('prefix', False):
                    topk_token = utils.filter_prefixes(topk_token)
                
                next_words[result[0]] = topk_token

            
            new_word, _ = self.calculate_scores(next_words, args)

            # 输出运行日志（可选参数）
            if args['log-info']:
                print(f"next_words:{next_words}")
                print(f"new_word:{new_word}")
                # print(f"args:{return_args}")
                
            # 把文本添加到模板中
            for i, text in enumerate(texts):
                texts[i] += new_word

            # 把文本添加到答案中
            ans += new_word

        return ans
    
    def call_app(self, text, info, extra_args):
        """
        调用单个模型的预测服务。

        :param text: 输入文本
        :param port: 服务端口
        :param choice: 模型编号
        :return: (choice, 模型预测结果)
        """
        
        # 默认使用transformers推理架构
        url = f"{info['model_url']}/predict"
        headers = {'Content-Type': 'application/json'}
        data = {'text': text, 'args': extra_args}
        response = requests.post(url, json=data, headers=headers)



        # 若非transformers架构，采用openai调用接口
        if info['model_arch'] != "transformers":
            result = {'prediction_values': [], 'args': {},  'sample_result': []}
            # 读取openai配置文件
            res_data = response.json()
            if res_data['response'][0]['bytes'] == []:
                res_data['response'][0]['token'] = info['EOS']
            result['sample_result'].append([res_data['response'][0]['token'], res_data['response'][0]['logprob']])

            for item in res_data['response'][0]['top_logprobs']:
                token = item['token']
                logprob = item['logprob']
                bytes = item['bytes']
                prob = math.exp(logprob)
                if  bytes == []:
                    token = info['EOS']
                result['prediction_values'].append([token, prob])

            return info["model_name"], result
        

        if response.status_code == 200:
            result = response.json()
            return info["model_name"], result
        else:
            print(f"{info['model_name']}, Error: {response.status_code}")
            return info["model_name"], []
        

        

    def call_template(self, question, info):
        """
        调用单个模型的模板生成服务。

        :param question: 输入问题
        :param port: 服务端口
        :return: 模板文本
        """

        # 若为transformers架构，则调用模板生成
        url = f"{info['model_url']}/template"
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
    
def process_chunk(choice):
    """
    输入：一个模型返回的 chunk 中的 CompletionChoice（包含 logprobs）
    输出：一个处理后的字典，包含：
        - 当前生成的 token 及其概率
        - 所有 top_logprobs 及其概率
    """

    # pdb.set_trace()  # 如果还需要调试可以开启

    # 获取当前 token 和其 logprob
    # pdb.set_trace()
    current_token = choice.logprobs.content[0]['token'] # 当前生成的 token
    current_logprob = choice.logprobs.content[0]['logprob']  # 获取当前 token 的 logprob
    current_prob = math.exp(current_logprob)  # 转换为概率

    # 获取当前 token 的 top_logprobs 列表（最多 N 个可能的候选 token）
    top_logprobs = choice.logprobs.content[0]['top_logprobs']

    # 转换为 [(token, logprob, prob), ...] 格式，并按 prob 排序
    alternatives = []
    for item in top_logprobs:
        token = item['token']
        logprob = item['logprob']
        prob = math.exp(logprob)
        alternatives.append((token, logprob, prob))

    # 按照概率从高到低排序
    alternatives.sort(key=lambda x: x[2], reverse=True)

    return {
        "current_token": current_token,
        "current_logprob": current_logprob,
        "current_probability": current_prob,
        "alternatives": alternatives  # 已排序
    }


# 使用示例
if __name__ == '__main__':
    # 0：Qwen1.5-7B-Chat
    # 2：Qwen2.5-7B-Instruct
    # 3：GLM-4-9B-Chat
    # 4：Meta-Llama-3.1-8B-Instruct

    # 实例化处理器
    handler = MultiModelHandler()

    # 选数据库中第i号模型
    model_choice = [3]

    question = "你知道智谱团队吗？"
    args = {
        'max_len': 500,
        'top_k': 5,
        'prefix': False,
        'soft': True,
        'log-info': False,
        'do_sample': True,
        'max_new_tokens':10,
        'temperature': 0.8,
        'return_dict_in_generate':True,
        # 'return_dict_in_generate':False,
        'output_scores': True,
        # 'output_scores': False,
        'top_p': 1,
        'handel_next_token': True,
        'mode': 0
    }

    # 调用生成答案
    result = handler.generate_response(model_choice, question, args)
    print(f"回答: {result}")
