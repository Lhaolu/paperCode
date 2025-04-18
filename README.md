# 运行说明

## 文件说明
```bash
paperCode/
├── fedcet_main.py        # 主脚本，包含实验运行逻辑
├── fedcet.py             # FedCET 算法实现
├── rfedcet.py            # RFedCET 算法实现
├── dst.py                # Random 算法与FedDST 算法实现
├── models.py             # 模型结构实现
├── datasets.py           # 数据集
├── prune.py              # 剪枝方法
├── README.md             # 项目说明文档
├── data/                 # 数据集存储目录
├── results/              # 实验结果保存目录
└── requirements.txt      # 依赖列表
```

项目包含以下四个核心文件：

1. **fedcet_main.py**：主程序文件，负责解析命令行参数、加载数据集、分配数据、创建模型并运行 FedCET 或 RFedCET 实验。
2. **fedcet.py**：实现 FedCET 算法的核心逻辑，包括客户端分组、稀疏模型初始化、参数探索和模型聚合。
3. **rfedcet.py**：实现 R-FedCET 算法，扩展了 FedCET，增加了超网络参数补偿和 UCB 稀疏度调整策略。
4. **dst.py**: 实现Random mask随机掩码算法与 FedDST算法 的核心逻辑。

## 安装依赖
```bash
      Python 3.8+
      PyTorch 1.9.0+
      torchvision
      NumPy
      scikit-learn
      matplotlib
      tqdm
```

```bash
      pip install -r requirements.txt
```

## 运行

#### 示例 1：运行 FedCET 算法（MNIST 数据集，CNN模型， IID）

```bash
python fedcet_main.py --algorithm fedcet --dataset mnist --model cnn --num_rounds 200 --sparsity 0.9 --num_clients 100 --local_epochs 5
```

#### 示例 2：运行 RFedCET 算法（CIFAR-10 数据集，ResNet18 模型，Non-IID）

```bash
python fedcet_main.py --algorithm rfedcet --dataset cifar10 --model resent18 --num_rounds 800 --sparsity 0.9 --num_clients 100 --local_epochs 5 --non_iid --dirichlet_alpha 0.5
```

#### 示例 3：运行 FedDST 算法（CIFAR-10 数据集，CNN 模型，Non-IID）

```bash
python dst.py --dataset cifar10 --sparsity 0.9 --readjustment-ratio 0.01 --rounds-between-readjustments 15
```

#### 示例 4：运行 Random算法（MNIST 数据集，CNN 模型，Non-IID）

```bash
python dst.py --dataset mnist --sparsity 0.8 --readjustment-ratio 0.0 --rounds 200
```
#### 参数说明

```bash
#通用参数
--algorithm：选择算法（fedcet 或 rfedcet）
--dataset：数据集（mnist、cifar10 或 cifar100）
--model：模型架构（cnn、mlp、vgg11 或 resnet18）
--num_rounds：通信轮次
--sparsity：目标稀疏度（0.0-1.0）
--num_clients：客户端总数
--clients_per_round：每轮参与的客户端数
--local_epochs：本地训练轮次
--non_iid：异构数据分布
--dirichlet_alpha：非 IID 数据分布的 Dirichlet 参数
--device：运行设备（cuda、mps 或 cpu）
--results_dir：结果保存目录（默认 ./results）
#R-FedCET参数
--loss_rate 丢包率
--embed_dim 嵌入向量维度
--change_ratio_init 参数转移集群数量
--param_ratio_init 参数转移数量
#FedDST参数
--rounds-between-readjustments 掩码更新间隔轮次
--readjustment-ratio 0.0 掩码更新调整比例
```
## 结果展示
实验结果会自动保存在 `results/`目录下的子目录中，每个实验目录以`{exp_name}_{algorithm}_{timestamp}`命名。以下是结果的结构和展示方式：
### 结果目录结构
每个实验目录包含以下文件：
```bash
results/my_experiment_rfedcet_20250418_123456/
├── config.json           # 实验配置参数
├── results.json          # 训练结果（包括准确率历史、最佳准确率等）
├── accuracy_plot.png     # 准确率随轮次变化的折线图
├── best_model.pth        # 最佳模型权重（如果设置了 --save_model）
├── final_model.pth       # 最终模型权重（如果设置了 --save_model）
```
+ config.json：记录所有实验参数，便于复现实验。
+ results.json：包含以下字段：
+ accuracy_history：每轮的测试准确率列表。
+ best_accuracy：训练过程中的最高准确率。
+ final_accuracy：最后一轮的准确率。
+ accuracy_plot.png：可视化准确率随通信轮次的变化，横轴为轮次，纵轴为准确率。
### 可视化结果
准确率曲线：accuracy_plot.png 显示准确率随轮次的变化，帮助分析模型的收敛性和稳定性。 示例：
比较多个实验： 使用 `fedcet_main.py` 中的 `compare_experiments` 函数比较多个实验的准确率曲线：
```bash
from fedcet_main import compare_experiments
exp_dirs = [
    "results/fedcet_mnist_20250418_123456",
    "results/rfedcet_mnist_20250418_123457"
]
labels = ["FedCET", "RFedCET"]
compare_experiments(exp_dirs, labels)
```
输出：comparison_plot.png，显示多个实验的准确率曲线对比。
