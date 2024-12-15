import os
import json
import matplotlib.pyplot as plt

# 假设你的 JSON 文件存储在一个文件夹中
folder_path = '/Users/sfeng/Documents/CMSC331_333/flowsim/draw'

# 初始化数据存储
hit_rates = []
quality_scores = []
retrieval_times = []
sender_strategies = []

i = 0
j= 0 
C = ['blue', 'red', 'gray', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# 遍历文件夹中的每个 JSON 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        # 读取 JSON 文件
        with open(os.path.join(folder_path, filename), 'r') as f:
            data = json.load(f)
        
        # 提取数据
        miss_rate = data['miss_rate']
        hit_rate = 1 - miss_rate
        quality_score = data['quality_score']
        retrieval_time = data['retrieval_time']
        sender_strategy = data['sender_strategy']
        
        # 保存数据
        hit_rates.append(hit_rate)
        quality_scores.append(quality_score)
        retrieval_times.append(retrieval_time)
        sender_strategies.append(sender_strategy)

# 绘制 hit rate vs. quality score
plt.figure(figsize=(10, 6))
for strategy in set(sender_strategies):  # 遍历不同的 sender_strategy
    # 筛选出当前 sender_strategy 的数据
    indices = [i for i, s in enumerate(sender_strategies) if s == strategy]
    plt.scatter([hit_rates[i] for i in indices], [quality_scores[i] for i in indices], label=strategy, color=C[i])
    i = i + 1

plt.xlabel('Hit Rate (1 - Miss Rate)')
plt.ylabel('Quality Score')
plt.title('Hit Rate vs Quality Score')
plt.legend()
plt.grid(True)
plt.ylim(0.8, 1)
plt.show()

# 绘制 hit rate vs. retrieval time
plt.figure(figsize=(10, 6))
for strategy in set(sender_strategies):
    if strategy != 'sjf_random':
        # 筛选出当前 sender_strategy 的数据
        indices = [i for i, s in enumerate(sender_strategies) if s == strategy]
        plt.scatter([hit_rates[i] for i in indices], [retrieval_times[i] for i in indices], label=strategy, color=C[j])
        j = j + 1

plt.xlabel('Hit Rate (1 - Miss Rate)')
plt.ylabel('Retrieval Time')
plt.title('Hit Rate vs Retrieval Time')
plt.legend()
plt.grid(True)
plt.show()
