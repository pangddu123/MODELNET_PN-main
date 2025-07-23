# MODELNET_PN
## 概述
MODELNET_PN 是一个用于生成响应的并联网络模型。通过加载多个模型并结合其输出，用户可以获取更为准确和多样化的生成结果。

## 使用方法
请按照以下步骤使用 MODELNET_PN：

### 1.添加模块路径 在您的 Python 脚本中，添加模型所在的路径。

```python
import sys
sys.path.append('/mnt/Data/xao/Parallel_Net/MODELNET_PN')
```

### 2.初始化并联网络 从 PN 模块中导入 MultiModelHandler 类并实例化。

```python
from PN import MultiModelHandler
handler = MultiModelHandler()
```

### 3.设置参数 配置生成响应所需的参数。以下是可用的参数及其说明：

```python
    args = {
        'max_len': 500,             # 生成响应的最大长度，单位是 token
        'topk': 3,                  # 在生成时选择的 top k 个候选结果
        'prefix': False,            # 是否使用前缀筛选（True/False）
        'soft': False,              # 是否使用softmax进行归一化（True/False）
        'log-info': True,
        'do_sample': True,
        'max_new_tokens':1,         #生成长度（不过貌似没用）
        'temperature':1,            #温度（不过貌似没用）
        'topp':None,                 # 在生成时选择的 top p 参数
        'mode': 0                   # 模式设置（0为计分制，1为投票制）
    }
```

### 4.选择模型并生成响应 使用 generate_response 方法生成响应，选择您想要使用的模型。

```python
model_choice = [9]  # 选择使用的模型ID，可以选择多个模型，例如：[1, 2, 3]
prompt = "您的输入提示"  # 替换为实际的输入提示
result = handler.generate_response(model_choice, prompt, args)
```