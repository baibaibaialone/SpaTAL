# SpaTAL

SpaTAL 用于 DLPFC 空间转录组的多切片空间域识别。当前项目包含四个切片的完整主流程，以及已经保存的实验结果。

| 数据集 | 切片 | 聚类数 |
| --- | --- | --- |
| DLPFC | 151673、151674、151675、151676 | 7 |

处理流程：**表达预处理 → ICP 空间配准 → 三维 KNN 构图 → 双视图对比训练 → mclust 聚类 → 评估与可视化**。

## 已保存的结果

`output/` 保存 **32 张 DLPFC 结果图片**和一份包含 **17 次 DLPFC 运行记录**的 `results.log`。历史图片、文件名和各次运行的数值均予以保留。

最近一条历史记录对应 `20260909_025938` 这一组图片，日志中的 `mean_ARI = 0.6618`、`mean_ACC = 0.7156`。

| 切片 | ARI | ACC¹ | NMI | DIS² |
| --- | ---: | ---: | ---: | ---: |
| 151673 | 0.675 | 0.728 | 0.728 | 0.0444 |
| 151674 | 0.755 | 0.772 | 0.771 | 0.0445 |
| 151675 | 0.641 | 0.706 | 0.706 | 0.0439 |
| 151676 | 0.576 | 0.656 | 0.656 | 0.0430 |

下图从左至右为上述四个切片的聚类空间散点图：

![DLPFC 四切片聚类结果](output/figure_20260909_025939.png)

可直接查看同一次运行的 [聚类 UMAP](output/umap_mclust_20260909_025938.png)、[真值 UMAP](output/umap_ground_truth_20260909_025938.png)、[切片 UMAP](output/umap_batch_name_20260909_025938.png) 和 [完整运行日志](output/results.log)。

这些文件来自原项目的历史 DLPFC 实验，本次恢复没有重新训练。日志保留原始标题、记录时间及当时的服务器路径；查看当前图片时使用本项目 `output/` 中的同名文件。

## 项目结构

```text
SpaTAL/
├── main.py                 # DLPFC 数据加载、配准、构图和实验入口
├── requirements.txt        # Python 直接依赖
├── spatal/
│   ├── config.py           # 训练与预处理参数
│   ├── model.py            # 图注意力编码器与对比损失
│   ├── attention.py        # 图注意力层
│   ├── learner.py          # 可学习邻接图
│   ├── trainer.py          # 训练和聚类评估
│   ├── preprocess.py       # AnnData 到 PyTorch 的数据转换
│   ├── evaluation.py       # 评估调度与内存中的最佳状态恢复
│   ├── metrics.py          # 标签处理、逐切片指标与结果输出
│   ├── utils.py            # 随机种子和 R/mclust 接口
│   └── __init__.py
└── output/                 # DLPFC 历史结果及后续运行输出
```

## 数据准备

原始表达数据没有随代码复制到本地。将四个切片放在同一个数据根目录下；每个切片采用以下结构：

```text
DLPFC/
├── 151673/
│   ├── 151673_filtered_feature_bc_matrix.h5
│   ├── 151673_truth.txt
│   └── spatial/
│       ├── tissue_positions_list.csv
│       ├── scalefactors_json.json
│       ├── tissue_hires_image.png
│       └── tissue_lowres_image.png
├── 151674/                 # 同样结构，文件前缀改为对应切片编号
├── 151675/
└── 151676/
```

`<切片编号>_truth.txt` 是**无表头、以制表符分隔的两列文件**：第一列为 spot barcode，第二列为真实标签。barcode 需要覆盖表达矩阵中的 spot。评估时过滤缺失值和 `unknown` 标签。

## 安装与运行

已验证的计算环境使用 Python 3.9。除 Python 依赖外，还需要可由 `rpy2` 调用的 R，以及 R 包 `mclust`。

```bash
cd ~/Desktop/SpaTAL
python -m pip install -r requirements.txt
R -e 'install.packages("mclust", repos="https://cloud.r-project.org")'
```

在已安装好依赖的环境中启动主程序：

```bash
cd ~/Desktop/SpaTAL
DLPFC_DATA_ROOT=/你的数据路径/DLPFC MPLBACKEND=Agg python main.py
```

| 环境变量 | 用途 | 默认值 |
| --- | --- | --- |
| `DLPFC_DATA_ROOT` | 四个切片所在的数据根目录 | `/tmp/baixu/DEEPST/data/DLPFC`，为原服务器路径 |
| `DLPFC_DEVICE` | PyTorch 计算设备 | 有 CUDA 时为 `cuda:0`，否则为 `cpu` |
| `MPLBACKEND` | Matplotlib 绘图后端 | 示例使用 `Agg`，直接保存图片 |

例如，需要使用 CPU 时，在运行命令前增加 `DLPFC_DEVICE=cpu`。使用 GPU 时，PyTorch 需要安装与计算环境匹配的 CUDA 构建。

## 参数设置

常用参数在 `spatal/config.py` 中修改，主程序直接读取 `Config`，不通过命令行参数覆盖这些设置。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `seed` | 2023 | 训练随机种子 |
| `top_genes` | 2000 | 每个切片筛选的高变基因数 |
| `k_cutoff` | 23 | 三维空间图的 KNN 邻居数 |
| `nps` | 30 | 聚类前对模型表示进行 PCA 的维数 |
| `learning_rate` | 0.001 | 模型和图学习器的学习率 |
| `weight_decay` | 0.0002 | 优化器权重衰减 |
| `gradient_clipping` | 5.0 | 模型梯度裁剪阈值 |
| `max_epoch` | 1650 | 训练轮数 |

以下设置位于具体实现中：

| 设置 | 当前值 | 位置 |
| --- | --- | --- |
| 切片与聚类数 | 四个切片、7 类 | `main.py` 的 `main()` |
| 输入特征 | 合并后 PCA 到 200 维 | `main.py` 的 `load_adata()` |
| 相邻切片间距 | 10、10、10 μm | `main.py` 的 `load_adata()` |
| 评估间隔 | 2000 轮，最后一轮强制评估 | `main.py` 的 `train_one()` |
| 编码器 | 3 个 256 维注意力头 | `spatal/model.py` |
| 图学习器 | 4 层、k 参数为 15 | `spatal/trainer.py` |
| 特征遮蔽 | 两个视图各遮蔽 20% 特征维度 | `spatal/trainer.py` 的 `loss_gcl()` |
| 基因图 | TMSB10 | `main.py` 的 `gene_list` |

在默认 1650 轮训练下，仅在最后一轮评估。保持 1650 轮时，可将评估间隔改为小于 1650 的正整数，以进行周期评估。

## 输出说明

所有新结果写入本项目的 `output/`；图片按生成时的时间戳命名，日志按运行追加。已有历史结果保留在同一目录中。

| 文件 | 内容 |
| --- | --- |
| `results.log` | 每次运行的参数、最佳评估轮次、逐切片指标和图片路径 |
| `figure_<时间>.png` | 四个切片的聚类空间散点图与各自 ARI |
| `umap_ground_truth_<时间>.png` | 按真实标签着色的 UMAP |
| `umap_mclust_<时间>.png` | 按聚类标签着色的 UMAP |
| `umap_batch_name_<时间>.png` | 按切片编号着色的 UMAP |
| `gene_TMSB10_<切片>_<时间>.png` | TMSB10 的空间表达图，基因在保留特征中时生成 |

历史目录中还保留了早期合并版 `umap_<时间>.png`。基因图使用当前预处理后的表达矩阵，不应解释为原始计数。当前程序自动保存 PNG 和日志；模型权重、嵌入矩阵及 `.h5ad` 不会自动导出。

¹ **ACC** 沿用项目原定义，为 `(NMI + homogeneity + completeness) / 3`，不代表分类正确率。`mean_ARI` 和 `mean_ACC` 为四个切片得分的等权平均。

² **DIS** 沿用 `(CHAOS + PAS) / 2`，与 ARI、NMI 分开报告。

当前精简版本已完成核心数值差分校验和合成数据两轮主流程测试，尚未重新运行完整 DLPFC 实验。由于删除未参与前向计算的随机初始化层，相同 seed 不保证逐值复现历史分数。

## 实现来源

模型实现参考 Min、Fang、Chen、Zhang（2025），[DOI: 10.1371/journal.pcbi.1012881](https://doi.org/10.1371/journal.pcbi.1012881)。ICP 配准参考 [ClayFlannigan/icp](https://github.com/ClayFlannigan/icp)。
