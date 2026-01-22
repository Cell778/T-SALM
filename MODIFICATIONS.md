# 主要修改说明 (Modifications Summary)

---

## 1. 时序建模 (Temporal Modeling with Transformer)

### 文件变更
- **`src/model/sclap.py`**：新增 `TemporalAudioEncoder` 类

### 核心思想
传统 CLAP 对整个音频进行全局池化，丢失了**时间维度的结构信息**。本修改通过以下方式保留时序特征：

1. **分块提取**：将特征图的时间轴（T）分成 4 个相等的块
   ```python
   T = sed_feature_maps.shape[2]
   chunck = T // 4
   
   # 对每个块的特征进行平均池化
   sed_chunck1 = sed_feature_maps[:,:,:chunck,:].mean(dim=(2,3))  # (B, C)
   sed_chunck2 = sed_feature_maps[:,:,chunck:2*chunck,:].mean(dim=(2,3))
   # ... chunck3, chunck4
   ```

2. **序列编码**：用 Transformer Encoder 建模块之间的时序依赖
   ```python
   temporal_seq = torch.stack([chunck1_embeds, chunck2_embeds, chunck3_embeds, chunck4_embeds], dim=1)  # (B, 4, D)
   audio_temporal_embeds = self.audio_temporal_encoder(temporal_seq)  # (B, D)
   ```

3. **联合投影**：将全局特征与时序特征拼接并投影
   ```python
   combined_audio_embeds = torch.cat([audio_embeds, audio_temporal_embeds], dim=-1)  # (B, 2D)
   audio_triplet_embeds = self.final_audio_projection(combined_audio_embeds)  # (B, D)
   ```

### 优势
- 捕捉**时序关系**（如"先A后B"的事件顺序）
- 在 Triplet Loss 中提供**更丰富的上下文**
- 对时间敏感的音频描述有更好的理解

### 代码位置
- 模型定义：`src/model/sclap.py:282-313` (TemporalAudioEncoder 类)
- 特征生成：`src/model/sclap.py:369-385` (get_audio_embedding 方法)

---

## 2. 3-Way Triplet Loss (TS Loss)

### 文件变更
- **`src/loss/loss_sclap.py`**：新增 `loss_logit_ts` 计算逻辑

### 核心思想
标准对比学习（InfoNCE）只利用 batch 内的随机负样本。本修改显式挖掘**硬负样本**：

#### 2-Way 对比（旧）
$$\text{Loss}_{2way} = -\log \frac{e^{\text{sim}(a, t_+)}}{e^{\text{sim}(a, t_+)} + \sum_{i=1}^{N-1} e^{\text{sim}(a, t_-^i)}}$$

#### 3-Way 对比（新）
$$\text{Loss}_{3way} = -\log \frac{e^{\text{sim}(a, t_+)}}{e^{\text{sim}(a, t_+)} + e^{\text{sim}(a, t_{neg\_t})} + e^{\text{sim}(a, t_{neg\_s})}}$$

其中：
- $t_+$ ：正样本文本（与音频配对的原始文本）
- $t_{neg\_t}$ ：**时间负样本**（同音频但来自不同时间的文本，即Triplet数据中的 neg_t）
- $t_{neg\_s}$ ：**空间负样本**（同音频但来自不同空间位置的文本，即Triplet数据中的 neg_s）

### 实现细节
```python
# 在 triplet_mode 下（stClotho 数据集）
pos_idx = torch.arange(b, device=device, dtype=torch.long) * g  # 每3个样本一组，取第0个为正
neg_t_idx = pos_idx + 1  # 负时序样本：第1个
neg_s_idx = pos_idx + 2  # 负空间样本：第2个

# 计算相似度
sim_pos = torch.sum(text_anchor * audio_pos, dim=-1)
sim_neg_t = torch.sum(text_anchor * audio_neg_t, dim=-1)
sim_neg_s = torch.sum(text_anchor * audio_neg_s, dim=-1)

# 构建3-way logits
logits_3way_t2a = logit_scale * torch.stack([sim_pos, sim_neg_t, sim_neg_s], dim=1)
labels_3way_t2a = torch.zeros(b, device=device, dtype=torch.long)  # 正样本为类别0
loss_logit_3way_t2a = F.cross_entropy(logits_3way_t2a, labels_3way_t2a)
```

### 优势
- **硬负样本挖掘**：不依赖随机的 batch 内负样本，而是显式构造时间和空间错误的负样本
- **更强的判别性**：模型必须学会区分"时间顺序错误"和"空间位置错误"这两种细粒度的不匹配
- **直接优化**：Loss 完全针对模型的弱点（混淆时间/空间）进行优化

### 代码位置
- 损失计算：`src/loss/loss_sclap.py:366-401` (3-way loss 计算)

### 配置
在 `configs/experiment/tSALM_triple.yaml` 中设置权重：
```yaml
loss_weights: [w_doa, w_semantic, w_temporal, w_spatial]  # [1.0, 0.5, 0.2, 0.2]
```

---

## 3. 模态分类器 (Modality Classifier with Gradient Reversal)

### 文件变更
- **`src/model/sclap.py`**：新增 `GradientReversalFunction` 和 `ModalityClassifier` 类
- **`src/model/model_module.py`**：新增模态分类器的前向传播和 Loss 计算

### 核心思想
**域对抗训练 (Domain Adversarial Training)**：让音频和文本的嵌入在共享表示空间中模态不可分。

#### Gradient Reversal Layer (GRL)
```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)  # 前向：恒等变换

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_  # 反向：取反 × λ
        return output, None
```

**关键机制**：
- **前向传播**：`output = input`（什么都不做）
- **反向传播**：`grad = -λ × grad_output`（梯度反向且缩放）

#### 模态分类器架构
```python
class ModalityClassifier(nn.Module):
    def __init__(self, input_dim):
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 8),
            nn.ReLU(),
            nn.Linear(input_dim // 8, input_dim // 16),
            nn.ReLU(),
            nn.Linear(input_dim // 16, 1),
            nn.Sigmoid()  # 二分类 (0=Audio, 1=Text)
        )
    
    def forward(self, x, lambda_=1.0):
        x = GradientReversalFunction.apply(x, lambda_)
        return self.classifier(x)
```

### 训练动态
1. **分类器优化**：最小化 BCE Loss，试图区分音频 vs 文本
   - 分类器训练：最小化 $\mathcal{L}_{cls}$
   
2. **Encoder 对抗**（经由 GRL）：最大化 BCE Loss，让音频和文本无法区分
   - Encoder 训练：最大化 $\mathcal{L}_{cls}$（梯度反向）

3. **结果**：Encoder 学会提取**模态无关的特征**，音频和文本在同一空间中对齐

### 代码位置
- GRL 定义：`src/model/sclap.py:14-23`
- 分类器定义：`src/model/sclap.py:25-43`
- 前向调用：`src/model/model_module.py:259-273`

---

## 4. Modality Loss 集成

### 文件变更
- **`src/model/model_module.py`**：在 `sCLAPModelModule.training_step()` 中新增模态 Loss 计算

### 实现细节

#### 数据准备
```python
# 提取主要对比特征
audio_emb = audio_features[-1]  # audio_triplet_embeds (B, D)
text_emb = text_features[0]      # text_comb_embeds (B, D)

# 沿 batch 维度拼接
modality_input = torch.cat([audio_emb, text_emb], dim=0)  # (2B, D)

# 构建标签：Audio=0, Text=1
labels_audio = torch.zeros(batch_size, 1, device=self.device)
labels_text = torch.ones(batch_size, 1, device=self.device)
modality_labels = torch.cat([labels_audio, labels_text], dim=0)  # (2B, 1)
```

#### 损失计算
```python
# 前向传播（含梯度反转）
modality_preds = self.net.modality_classifier(modality_input)  # (2B, 1)

# 二分类交叉熵
modality_loss = F.binary_cross_entropy(modality_preds, modality_labels)

# 加权系数（从配置读取）
w_modality = loss_weights[3]  # 默认 0.2

# 加入总损失
total_loss['total_loss'] += w_modality * modality_loss
```

### 总损失函数
$$\mathcal{L}_{total} = w_{ss} L_{spatial\_semantic} + w_{sem} L_{semantic} + w_{doa} L_{doa} + w_{ts} L_{ts} + w_{mod} L_{modality}$$

其中：
- $w_{ss} = 1 - w_{sem}^{eff}$ ：空间-语义权重
- $w_{sem}^{eff}$ ：从 epoch 3 开始逐步增加（ramp-up）
- $w_{ts}^{eff}$ ：从 epoch 0 开始逐步增加（ramp-up）
- $w_{mod} = 0.2$ ：模态 Loss 权重（可配置）

### 代码位置
- 模态 Loss 计算：`src/model/model_module.py:259-273`
- Loss 集成：`src/model/model_module.py:302-303`
- 指标记录：`src/model/model_module.py:70` (初始化) 和 `src/model/model_module.py:304-306` (更新)

---

## 配置参数

### `configs/experiment/tSALM_triple.yaml`
```yaml
model:
  loss_weights: [1.0, 0.5, 0.2, 0.2]  # [w_doa, w_sem, w_ts, w_modality]
```

### 内部参数（在 Loss 函数中自动调整）
```python
# 语义 Loss 权重：第3个 epoch 后启用
w_sem_eff = 0.0 if epoch_it < 3 else w_sem

# 时序 Loss 权重：前4个 epoch 线性 ramp-up
temporal_ramp_epochs = 4
progress = float(epoch_it + 1) / float(temporal_ramp_epochs)
w_temp_eff = w_temp * min(1.0, progress)
```

---

## 预期效果

| 指标 | 改进原因 |
|------|--------|
| **mAP@1** | 时序建模捕捉事件顺序；3-way Loss 提供硬负样本；模态对齐减少 Gap |
| **mAP@10** | 3-way Loss 的细粒度判别性；模态分类器的正则化效应 |
| **Recall@1** | 更强的特征表示；模态无关特征的共享空间 |
| **Temporal Sensitivity** | 明确的时序建模；neg_t 样本的显式优化 |

---

## 调试建议

### 1. 检查 Loss 趋势
```python
# 监控各分量 Loss
- loss_logit_ts        # 应该快速下降
- loss_modality        # 应该在 0.5 附近震荡（二分类 Loss）
- total_loss           # 总体下降趋势
```

### 2. 梯度反转调试
如果 `loss_modality` 不下降，尝试：
```yaml
# 降低权重
loss_weights: [1.0, 0.5, 0.2, 0.1]  # 改小最后一个

# 或减小 λ（在 ModalityClassifier 中硬编码）
lambda_ = 0.5  # 之前是 1.0
```

### 3. 过拟合防治
```python
# 增加 Dropout（TemporalAudioEncoder 中已有 0.1）
# 减小 ModalityClassifier 的隐层维度
# 增加数据增强
```

---

## 文件变更汇总

| 文件 | 变更 | 行号 |
|------|------|------|
| `src/model/sclap.py` | 新增 GradientReversalFunction | 14-23 |
| `src/model/sclap.py` | 新增 ModalityClassifier | 25-43 |
| `src/model/sclap.py` | 新增 TemporalAudioEncoder | 282-313 |
| `src/model/sclap.py` | 修改 sCLAP_Dual.__init__ | 添加 modality_classifier |
| `src/model/sclap.py` | 修改 get_audio_embedding | 369-385 |
| `src/loss/loss_sclap.py` | 新增 3-way Triplet Loss | 366-401 |
| `src/model/model_module.py` | 新增 modality Loss 计算 | 259-273 |
| `src/model/model_module.py` | 修改 training_step | 302-303 |
| `configs/experiment/tSALM_triple.yaml` | 新增 loss_weights[3] | 25 |

---

**更新日期**：2026 年 1 月 22 日  
**作者**：Cell 
