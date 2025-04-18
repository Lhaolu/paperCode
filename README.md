# 运行说明

## 文件说明

项目包含以下三个核心文件：

1. **fedcet_main.py**：主程序文件，负责解析命令行参数、加载数据集、分配数据、创建模型并运行 FedCET 或 RFedCET 实验。
2. **fedcet.py**：实现 FedCET 算法的核心逻辑，包括客户端分组、稀疏模型初始化、参数探索和模型聚合。
3. **rfedcet.py**：实现 R-FedCET 算法，扩展了 FedCET，增加了超网络参数补偿和 UCB 稀疏度调整策略。
4. **dst.py**: 实现Random mask随机掩码算法与 FedDST算法 的核心逻辑。

## 安装依赖

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



