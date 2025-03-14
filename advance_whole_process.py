import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import json
import numpy as np
import pickle
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import seaborn as sns

# 定义常量
AA_LETTERS = "-ACDEFGHIKLMNPQRSTVWYX"  # 添加填充符"-"和未知氨基酸"X"
MAX_LENGTH = 20  # TCR序列最大长度

# 创建氨基酸到索引的映射
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LETTERS)}

class TCRAggregator(nn.Module):
    """
    TCR库特征聚合器
    
    将同一库内的多个TCR序列特征聚合为单个库级别特征向量
    支持多种聚合策略：均值、最大值、注意力、多头注意力和自适应聚合
    """
    def __init__(self, feature_dim=128, aggregation_type="attention", num_heads=4):
        """
        初始化聚合器
        
        参数:
        - feature_dim: 输入特征维度
        - aggregation_type: 聚合类型，可选 'mean', 'max', 'attention', 'multi_head', 'adaptive'
        - num_heads: 多头注意力的头数
        """
        super(TCRAggregator, self).__init__()
        self.aggregation_type = aggregation_type
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        if aggregation_type == "attention":
            # 注意力聚合
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        elif aggregation_type == "multi_head":
            # 多头注意力聚合
            self.attention_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1)
                ) for _ in range(num_heads)
            ])
            self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
            self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        elif aggregation_type == "adaptive":
            # 自适应聚合 - 结合多种聚合方式
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            # 聚合方式权重
            self.agg_weights = nn.Parameter(torch.ones(3) / 3)  # [mean, max, attention]
            self.output_projection = nn.Linear(feature_dim, feature_dim)
        
        elif aggregation_type == "hierarchical":
            # 分层注意力聚合
            # 第一层：序列特征注意力
            self.seq_attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            # 第二层：子集特征注意力
            self.cluster_attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            # K-means聚类器用于序列分组
            self.n_clusters = 3  # 聚类数量
            self.output_projection = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, features, mask=None):
        """
        前向传播函数
        
        参数:
        - features: 形状为 [batch_size, feature_dim] 的特征向量
        - mask: 形状为 [num_libraries, batch_size] 的掩码矩阵
               如果序列j属于库i，则mask[i,j]=1，否则为0
        
        返回:
        - 形状为 [num_libraries, feature_dim] 的库级别特征向量
        """
        batch_size = features.shape[0]
        
        if mask is None:
            # 假设所有特征都属于同一个库
            mask = torch.ones(1, batch_size, device=features.device)
        
        num_libraries = mask.shape[0]
        library_features = torch.zeros(num_libraries, self.feature_dim, device=features.device)
        
        # 根据聚合类型选择相应的聚合方法
        if self.aggregation_type == "mean":
            # 简单平均聚合
            for i in range(num_libraries):
                if mask[i].sum() > 0:  # 确保库中有序列
                    library_features[i] = (features * mask[i].unsqueeze(1)).sum(0) / mask[i].sum()
        
        elif self.aggregation_type == "max":
            # 最大池化聚合
            for i in range(num_libraries):
                if mask[i].sum() > 0:
                    # 创建掩码版本的特征，将非该库的序列设为负无穷
                    masked_features = features.clone()
                    masked_features[mask[i] == 0] = float('-inf')
                    # 按维度取最大值
                    library_features[i] = torch.max(masked_features, dim=0)[0]
        
        elif self.aggregation_type == "attention":
            # 注意力聚合
            attention_weights = self.attention(features).squeeze(-1)  # [batch_size]
            
            for i in range(num_libraries):
                if mask[i].sum() > 0:
                    # 将不属于此库的序列的注意力设为负无穷
                    masked_attention = attention_weights.clone()
                    masked_attention[mask[i] == 0] = float('-inf')
                    
                    # Softmax获取归一化权重
                    weights = F.softmax(masked_attention, dim=0)
                    
                    # 加权聚合
                    library_features[i] = (features * weights.unsqueeze(1)).sum(0)
        
        elif self.aggregation_type == "multi_head":
            # 多头注意力聚合
            all_head_features = []
            
            for head in self.attention_heads:
                attention_weights = head(features).squeeze(-1)
                head_library_features = torch.zeros(num_libraries, self.feature_dim, device=features.device)
                
                for i in range(num_libraries):
                    if mask[i].sum() > 0:
                        masked_attention = attention_weights.clone()
                        masked_attention[mask[i] == 0] = float('-inf')
                        weights = F.softmax(masked_attention, dim=0)
                        head_library_features[i] = (features * weights.unsqueeze(1)).sum(0)
                
                all_head_features.append(head_library_features)
            
            # 融合多头结果
            head_weights_norm = F.softmax(self.head_weights, dim=0)
            for i, head_features in enumerate(all_head_features):
                library_features += head_features * head_weights_norm[i]
            
            # 最终投影
            library_features = self.output_projection(library_features)
        
        elif self.aggregation_type == "adaptive":
            # 自适应聚合 - 动态结合多种聚合策略
            mean_features = torch.zeros(num_libraries, self.feature_dim, device=features.device)
            max_features = torch.zeros(num_libraries, self.feature_dim, device=features.device)
            att_features = torch.zeros(num_libraries, self.feature_dim, device=features.device)
            
            # 计算注意力权重
            attention_weights = self.attention(features).squeeze(-1)
            
            for i in range(num_libraries):
                if mask[i].sum() > 0:
                    # 平均聚合
                    mean_features[i] = (features * mask[i].unsqueeze(1)).sum(0) / mask[i].sum()
                    
                    # 最大池化
                    masked_features = features.clone()
                    masked_features[mask[i] == 0] = float('-inf')
                    max_features[i] = torch.max(masked_features, dim=0)[0]
                    
                    # 注意力聚合
                    masked_attention = attention_weights.clone()
                    masked_attention[mask[i] == 0] = float('-inf')
                    weights = F.softmax(masked_attention, dim=0)
                    att_features[i] = (features * weights.unsqueeze(1)).sum(0)
            
            # 计算自适应权重
            agg_weights_norm = F.softmax(self.agg_weights, dim=0)
            
            # 加权组合不同聚合策略的结果
            library_features = (mean_features * agg_weights_norm[0] + 
                               max_features * agg_weights_norm[1] + 
                               att_features * agg_weights_norm[2])
            
            # 最终投影
            library_features = self.output_projection(library_features)
        
        elif self.aggregation_type == "hierarchical":
            # 分层注意力聚合
            for i in range(num_libraries):
                if mask[i].sum() > 0:
                    # 提取该库的序列特征
                    lib_mask = mask[i]
                    lib_features = features[lib_mask > 0]
                    
                    if len(lib_features) == 0:
                        continue
                    
                    if len(lib_features) <= self.n_clusters:
                        # 序列数少于聚类数，直接使用注意力机制
                        att_weights = self.seq_attention(lib_features).squeeze(-1)
                        weights = F.softmax(att_weights, dim=0)
                        library_features[i] = (lib_features * weights.unsqueeze(1)).sum(0)
                    else:
                        # 使用K-means将序列分组
                        with torch.no_grad():
                            # 转到CPU进行聚类
                            cpu_features = lib_features.detach().cpu().numpy()
                            from sklearn.cluster import KMeans
                            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
                            clusters = kmeans.fit_predict(cpu_features)
                            clusters = torch.tensor(clusters, device=lib_features.device)
                        
                        # 计算每个聚类的特征表示
                        cluster_features = []
                        for c in range(self.n_clusters):
                            cluster_mask = (clusters == c)
                            if cluster_mask.sum() > 0:
                                cluster_feats = lib_features[cluster_mask]
                                
                                # 第一层：计算聚类内序列的注意力
                                seq_att = self.seq_attention(cluster_feats).squeeze(-1)
                                seq_weights = F.softmax(seq_att, dim=0)
                                cluster_feat = (cluster_feats * seq_weights.unsqueeze(1)).sum(0)
                                cluster_features.append(cluster_feat)
                        
                        if cluster_features:
                            cluster_features = torch.stack(cluster_features)
                            
                            # 第二层：计算聚类间的注意力
                            cluster_att = self.cluster_attention(cluster_features).squeeze(-1)
                            cluster_weights = F.softmax(cluster_att, dim=0)
                            library_features[i] = (cluster_features * cluster_weights.unsqueeze(1)).sum(0)
                
                # 最终投影
                library_features = self.output_projection(library_features)
        
        return library_features


class TCRClassifier(nn.Module):
    """TCR库分类器(Cls)"""
    def __init__(self, input_dim=128, num_classes=2):
        super(TCRClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class TCRExtractor(nn.Module):
    """TCR序列的特征提取网络(Ext)"""
    def __init__(self, in_channels=32, feature_dim=128):
        super(TCRExtractor, self).__init__()
        
        # 多尺度卷积层，捕获不同长度的模式
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        
        # 池化后的特征融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 最终特征映射层
        self.fc = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        """
        参数:
        x: 形状为 [batch_size, seq_len, embed_dim] 的嵌入向量
        
        返回:
        形状为 [batch_size, feature_dim] 的特征向量
        """
        # 调整维度顺序以适应卷积 [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        # 连接多尺度特征
        x = torch.cat([x1, x2, x3], dim=1)
        
        # 特征融合
        x = self.fusion(x)
        
        # 全局最大池化
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        
        # 最终特征映射
        x = self.fc(x)
        
        return x


class TCREmbedding(nn.Module):
    """TCR序列的嵌入层(Emb)"""
    def __init__(self, vocab_size=len(AA_LETTERS), embed_dim=32):
        super(TCREmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
    def forward(self, x):
        """
        参数:
        x: 形状为 [batch_size, seq_len] 的整数张量
        
        返回:
        形状为 [batch_size, seq_len, embed_dim] 的嵌入向量
        """
        return self.embedding(x)


class TCRDataset(Dataset):
    """TCR序列数据集类"""
    def __init__(self, libraries, labels=None, max_length=MAX_LENGTH):
        self.libraries = libraries
        self.labels = labels
        self.max_length = max_length
        
        # 展平库，记录每个序列属于哪个库
        self.sequences = []
        self.library_ids = []
        for lib_id, library in enumerate(libraries):
            self.sequences.extend(library)
            self.library_ids.extend([lib_id] * len(library))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        lib_id = self.library_ids[idx]
        
        # 序列编码
        encoded_seq = self.encode_sequence(seq)
        
        if self.labels is not None:
            label = self.labels[lib_id]
            return encoded_seq, lib_id, label
        else:
            return encoded_seq, lib_id
    
    def encode_sequence(self, seq):
        """将氨基酸序列编码为整数索引序列，并进行填充"""
        # 将序列转换为整数索引
        encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:self.max_length]]
        
        # 填充序列
        padding_length = self.max_length - len(encoded)
        if padding_length > 0:
            encoded = encoded + [AA_TO_IDX['-']] * padding_length
            
        return torch.tensor(encoded, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        sequences, lib_ids, labels = zip(*batch)
        sequences = torch.stack(sequences)
        lib_ids = torch.tensor(lib_ids)
        labels = torch.tensor(labels)
        return sequences, lib_ids, labels

# 定义对比学习损失函数
class NTXentLoss(nn.Module):
    """归一化温度尺度交叉熵损失，用于对比学习"""
    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, features, lib_ids):
        """
        参数:
        features: 形状为 [batch_size, feature_dim] 的特征向量
        lib_ids: 形状为 [batch_size] 的库ID
        
        返回:
        对比学习损失
        """
        # 归一化特征向量
        features = F.normalize(features, dim=1)
        
        # 计算所有样本对之间的相似度
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签矩阵：如果两个样本来自同一库，则为正样本对
        labels = (lib_ids.unsqueeze(1) == lib_ids.unsqueeze(0)).float()
        
        # 移除自身相似度
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)
        
        # 正样本对的相似度应该比负样本对高
        loss = self.criterion(sim_matrix, labels)
        
        return loss / labels.shape[0]


def pattern_discovery_analysis(
    test_libraries, test_labels,
    embedding_model, extractor_model,
    class_patterns, device='cuda',
    save_dir='results/pattern_discovery'
):
    """
    分析模型是否成功识别出各类别的序列特征模式，并尝试发现潜在的新模式
    
    参数:
    - test_libraries: 测试集库
    - test_labels: 测试集标签
    - embedding_model: 氨基酸序列嵌入模型
    - extractor_model: 特征提取模型
    - class_patterns: 每个类别已知的特征模式
    - device: 使用的设备
    - save_dir: 保存分析结果的目录
    
    返回:
    - pattern_stats: 模式统计信息
    """
    print("\n开始模式发现与验证分析...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将模型设置为评估模式
    embedding_model.eval()
    extractor_model.eval()
    
    # 获取数据集中的类别
    unique_classes = sorted(set(test_labels))
    
    # 按类别组织库
    class_libraries = {label: [] for label in unique_classes}
    for lib, label in zip(test_libraries, test_labels):
        class_libraries[label].append(lib)
    
    # 按类别收集所有序列
    class_sequences = {label: [] for label in unique_classes}
    for label, libraries in class_libraries.items():
        for lib in libraries:
            class_sequences[label].extend(lib)
    
    # 1. 模式出现频率统计
    print("分析模式出现频率...")
    pattern_stats = {}
    
    # 对每个类别进行模式统计
    for label in unique_classes:
        pattern_stats[label] = {'patterns': {}, 'total_sequences': len(class_sequences[label])}
        
        # 统计每个模式出现的次数和频率
        for pattern in set([p for patterns in class_patterns.values() for p in patterns]):
            pattern_count = sum(1 for seq in class_sequences[label] if pattern in seq)
            pattern_freq = pattern_count / len(class_sequences[label]) if class_sequences[label] else 0
            pattern_stats[label]['patterns'][pattern] = {
                'count': pattern_count,
                'frequency': pattern_freq
            }
    
    # 将统计结果写入文件
    with open(os.path.join(save_dir, 'pattern_frequency.txt'), 'w') as f:
        f.write("模式出现频率统计\n")
        f.write("=================\n\n")
        
        for label in unique_classes:
            f.write(f"类别 {label} (共 {pattern_stats[label]['total_sequences']} 条序列):\n")
            f.write("-------------------------------------------------\n")
            
            # 按频率排序
            sorted_patterns = sorted(
                pattern_stats[label]['patterns'].items(),
                key=lambda x: x[1]['frequency'],
                reverse=True
            )
            
            for pattern, stats in sorted_patterns:
                is_class_pattern = pattern in class_patterns.get(label, [])
                class_marker = "*" if is_class_pattern else ""
                f.write(f"{pattern}{class_marker}: {stats['count']} 次 ({stats['frequency']:.2%})\n")
            
            f.write("\n")
        
        f.write("注: * 号标记的是该类别的已知特征模式\n")
    
    # 2. 模式在不同类别间的区分能力分析
    print("分析模式区分能力...")
    
    # 计算每个模式的类别鉴别力
    pattern_discrimination = {}
    all_patterns = set([p for patterns in class_patterns.values() for p in patterns])
    
    for pattern in all_patterns:
        pattern_discrimination[pattern] = {}
        for label in unique_classes:
            freq = pattern_stats[label]['patterns'][pattern]['frequency']
            pattern_discrimination[pattern][label] = freq
    
    # 可视化不同类别间的模式分布
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(all_patterns))
    width = 0.8 / len(unique_classes)
    patterns_list = list(all_patterns)
    
    for i, label in enumerate(unique_classes):
        freqs = [pattern_discrimination[pattern][label] for pattern in patterns_list]
        plt.bar(x + i*width - width*(len(unique_classes)-1)/2, freqs, width, 
               label=f'类别{label}')
    
    plt.xlabel('模式')
    plt.ylabel('出现频率')
    plt.title('不同类别中模式的出现频率')
    plt.xticks(x, patterns_list, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pattern_discrimination.png'), dpi=200)
    plt.close()
    
    # 3. 模式对特征表示的影响分析
    print("分析模式对特征表示的影响...")
    
    # 为每个类别和模式收集序列特征
    features_by_pattern = {label: {} for label in unique_classes}
    
    # 批处理大小
    batch_size = 128
    
    for label in unique_classes:
        # 对每个模式
        for pattern in all_patterns:
            # 收集包含该模式的序列
            with_pattern = [seq for seq in class_sequences[label] if pattern in seq]
            without_pattern = [seq for seq in class_sequences[label] if pattern not in seq]
            
            # 限制样本数量以加快处理速度
            max_samples = 100
            if len(with_pattern) > max_samples:
                with_pattern = random.sample(with_pattern, max_samples)
            if len(without_pattern) > max_samples:
                without_pattern = random.sample(without_pattern, max_samples)
            
            # 提取特征 - 包含模式的序列
            with_features = []
            for i in range(0, len(with_pattern), batch_size):
                batch = with_pattern[i:i+batch_size]
                encoded_batch = []
                for seq in batch:
                    encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                    padding_length = MAX_LENGTH - len(encoded)
                    if padding_length > 0:
                        encoded = encoded + [AA_TO_IDX['-']] * padding_length
                    encoded_batch.append(encoded)
                
                batch_tensor = torch.tensor(encoded_batch, dtype=torch.long).to(device)
                
                with torch.no_grad():
                    embeddings = embedding_model(batch_tensor)
                    features = extractor_model(embeddings).cpu().numpy()
                    with_features.extend(features)
            
            # 提取特征 - 不包含模式的序列
            without_features = []
            if without_pattern:  # 确保有不包含模式的序列
                for i in range(0, len(without_pattern), batch_size):
                    batch = without_pattern[i:i+batch_size]
                    encoded_batch = []
                    for seq in batch:
                        encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                        padding_length = MAX_LENGTH - len(encoded)
                        if padding_length > 0:
                            encoded = encoded + [AA_TO_IDX['-']] * padding_length
                        encoded_batch.append(encoded)
                    
                    batch_tensor = torch.tensor(encoded_batch, dtype=torch.long).to(device)
                    
                    with torch.no_grad():
                        embeddings = embedding_model(batch_tensor)
                        features = extractor_model(embeddings).cpu().numpy()
                        without_features.extend(features)
            
            # 保存特征
            features_by_pattern[label][pattern] = {
                'with_pattern': np.array(with_features) if with_features else np.array([]),
                'without_pattern': np.array(without_features) if without_features else np.array([]),
                'with_count': len(with_pattern),
                'without_count': len(without_pattern)
            }
    
    # 4. 模式特征显著性分析
    print("分析模式特征显著性...")
    
    # 计算每个模式的特征差异显著性
    pattern_significance = {}
    
    for label in unique_classes:
        pattern_significance[label] = {}
        for pattern in all_patterns:
            pattern_data = features_by_pattern[label][pattern]
            
            # 如果有足够的样本来做统计检验
            if (pattern_data['with_count'] > 5 and pattern_data['without_count'] > 5 and
                len(pattern_data['with_pattern']) > 0 and len(pattern_data['without_pattern']) > 0):
                
                # 计算特征向量的平均余弦相似度
                with_mean = np.mean(pattern_data['with_pattern'], axis=0)
                without_mean = np.mean(pattern_data['without_pattern'], axis=0)
                
                # 归一化向量
                with_norm = np.linalg.norm(with_mean)
                without_norm = np.linalg.norm(without_mean)
                
                if with_norm > 0 and without_norm > 0:
                    with_mean_normalized = with_mean / with_norm
                    without_mean_normalized = without_mean / without_norm
                    
                    # 余弦相似度
                    cosine_sim = np.dot(with_mean_normalized, without_mean_normalized)
                    
                    # 特征差异度 = 1 - 余弦相似度
                    feature_diff = 1.0 - cosine_sim
                else:
                    feature_diff = 0
                
                # 计算特征激活幅度的差异
                activation_diff = np.mean(np.abs(with_mean - without_mean))
                
                pattern_significance[label][pattern] = {
                    'feature_diff': feature_diff,
                    'activation_diff': activation_diff,
                    'is_class_pattern': pattern in class_patterns.get(label, [])
                }
            else:
                # 样本不足
                pattern_significance[label][pattern] = {
                    'feature_diff': 0,
                    'activation_diff': 0,
                    'is_class_pattern': pattern in class_patterns.get(label, [])
                }
    
    # 将结果写入文件
    with open(os.path.join(save_dir, 'pattern_significance.txt'), 'w') as f:
        f.write("模式特征显著性分析\n")
        f.write("=================\n\n")
        
        for label in unique_classes:
            f.write(f"类别 {label} 模式显著性:\n")
            f.write("-------------------------------------------------\n")
            
            # 按特征差异度排序
            sorted_patterns = sorted(
                pattern_significance[label].items(),
                key=lambda x: x[1]['feature_diff'],
                reverse=True
            )
            
            f.write("按特征方向差异排序:\n")
            for pattern, stats in sorted_patterns:
                is_class_pattern = stats['is_class_pattern']
                class_marker = "*" if is_class_pattern else ""
                f.write(f"{pattern}{class_marker}: 特征差异度 = {stats['feature_diff']:.4f}, "
                       f"激活差异 = {stats['activation_diff']:.4f}\n")
            
            f.write("\n")
        
        f.write("注: * 号标记的是该类别的已知特征模式\n")
    
    # 5. 可视化模式特征差异
    plt.figure(figsize=(12, len(unique_classes)*4))
    
    for i, label in enumerate(unique_classes):
        plt.subplot(len(unique_classes), 1, i+1)
        
        patterns = list(pattern_significance[label].keys())
        feature_diffs = [pattern_significance[label][p]['feature_diff'] for p in patterns]
        activation_diffs = [pattern_significance[label][p]['activation_diff'] * 5 for p in patterns]  # 缩放以便于可视化
        is_class_pattern = [pattern_significance[label][p]['is_class_pattern'] for p in patterns]
        
        # 按特征差异度排序
        sorted_indices = np.argsort(feature_diffs)[::-1]
        patterns = [patterns[i] for i in sorted_indices]
        feature_diffs = [feature_diffs[i] for i in sorted_indices]
        activation_diffs = [activation_diffs[i] for i in sorted_indices]
        is_class_pattern = [is_class_pattern[i] for i in sorted_indices]
        
        x = np.arange(len(patterns))
        
        # 使用不同颜色区分已知模式和其他模式
        bar_colors = ['green' if is_pattern else 'lightgray' for is_pattern in is_class_pattern]
        
        plt.bar(x, feature_diffs, width=0.4, color=bar_colors, 
               label='特征方向差异')
        plt.bar(x + 0.4, activation_diffs, width=0.4, color='orange', alpha=0.7,
               label='激活幅度差异(缩放)')
        
        plt.xlabel('模式')
        plt.ylabel('差异度')
        plt.title(f'类别{label}的模式特征显著性')
        plt.xticks(x + 0.2, patterns, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pattern_feature_significance.png'), dpi=200)
    plt.close()
    
    # 6. 模式预测能力分析 - 单模式序列分类实验
    print("分析单模式序列分类能力...")
    
    # 创建一个简单分类器，使用提取的特征进行分类
    simple_classifier = LinearSVC(random_state=42)
    
    # 收集训练数据 - 使用中等数量的序列
    max_seqs_per_class = 300
    X_train = []
    y_train = []
    
    for label in unique_classes:
        # 随机选择一些序列
        if len(class_sequences[label]) > max_seqs_per_class:
            selected_seqs = random.sample(class_sequences[label], max_seqs_per_class)
        else:
            selected_seqs = class_sequences[label]
        
        # 提取特征
        seq_features = []
        for i in range(0, len(selected_seqs), batch_size):
            batch = selected_seqs[i:i+batch_size]
            encoded_batch = []
            for seq in batch:
                encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                padding_length = MAX_LENGTH - len(encoded)
                if padding_length > 0:
                    encoded = encoded + [AA_TO_IDX['-']] * padding_length
                encoded_batch.append(encoded)
            
            batch_tensor = torch.tensor(encoded_batch, dtype=torch.long).to(device)
            
            with torch.no_grad():
                embeddings = embedding_model(batch_tensor)
                features = extractor_model(embeddings).cpu().numpy()
                seq_features.extend(features)
        
        X_train.extend(seq_features)
        y_train.extend([label] * len(seq_features))
    
    # 训练分类器
    simple_classifier.fit(X_train, y_train)
    
    # 生成合成序列并预测
    pattern_prediction_stats = {pattern: {'pred_counts': {}} for pattern in all_patterns}
    
    # 为每个模式创建合成序列
    for pattern in all_patterns:
        # 生成包含该模式的随机序列
        synth_seqs = []
        for _ in range(100):  # 生成100个序列
            # 随机生成序列长度
            seq_len = random.randint(10, 16)
            # 随机生成氨基酸序列
            seq = ''.join(random.choice(AA_LETTERS.replace('-', '').replace('X', '')) 
                         for _ in range(seq_len))
            
            # 在随机位置插入模式
            pos = random.randint(0, max(0, len(seq) - len(pattern)))
            seq = seq[:pos] + pattern + seq[pos+len(pattern):]
            if len(seq) > MAX_LENGTH:
                seq = seq[:MAX_LENGTH]
            
            synth_seqs.append(seq)
        
        # 提取特征
        synth_features = []
        for i in range(0, len(synth_seqs), batch_size):
            batch = synth_seqs[i:i+batch_size]
            encoded_batch = []
            for seq in batch:
                encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                padding_length = MAX_LENGTH - len(encoded)
                if padding_length > 0:
                    encoded = encoded + [AA_TO_IDX['-']] * padding_length
                encoded_batch.append(encoded)
            
            batch_tensor = torch.tensor(encoded_batch, dtype=torch.long).to(device)
            
            with torch.no_grad():
                embeddings = embedding_model(batch_tensor)
                features = extractor_model(embeddings).cpu().numpy()
                synth_features.extend(features)
        
        # 预测类别
        predictions = simple_classifier.predict(synth_features)
        
        # 统计预测结果
        for label in unique_classes:
            count = np.sum(predictions == label)
            pattern_prediction_stats[pattern]['pred_counts'][label] = count
        
        # 计算预测概率
        for label in unique_classes:
            count = pattern_prediction_stats[pattern]['pred_counts'][label]
            pattern_prediction_stats[pattern]['pred_prob'] = {
                label: count / len(predictions) for label in unique_classes
            }
    
    # 绘制模式预测结果
    plt.figure(figsize=(12, 8))
    
    x = np.arange(len(all_patterns))
    width = 0.8 / len(unique_classes)
    patterns_list = list(all_patterns)
    
    for i, label in enumerate(unique_classes):
        probs = [pattern_prediction_stats[pattern]['pred_counts'][label] / 100 
                for pattern in patterns_list]
        plt.bar(x + i*width - width*(len(unique_classes)-1)/2, probs, width, 
               label=f'预测为类别{label}')
    
    plt.xlabel('模式')
    plt.ylabel('预测概率')
    plt.title('单模式序列的分类预测结果')
    plt.xticks(x, patterns_list, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pattern_prediction.png'), dpi=200)
    plt.close()
    
    # 7. 综合评估每个模式的识别性能
    print("进行模式综合评估...")
    
    # 评估指标: 区分能力 + 特征差异 + 预测性能
    pattern_overall_score = {}
    
    for pattern in all_patterns:
        # 找出模式"属于"的类别 (原始指定或频率最高的类别)
        pattern_class = None
        max_freq = 0
        for c, patterns in class_patterns.items():
            if pattern in patterns:
                pattern_class = c
                break
        
        if pattern_class is None:
            # 如果不是已知的类别模式，则选择频率最高的类别
            for label in unique_classes:
                freq = pattern_stats[label]['patterns'][pattern]['frequency']
                if freq > max_freq:
                    max_freq = freq
                    pattern_class = label
        
        # 计算区分能力 - 该模式在"自己"类别与其他类别频率的差距
        discrimination_score = pattern_stats[pattern_class]['patterns'][pattern]['frequency']
        for label in unique_classes:
            if label != pattern_class:
                discrimination_score -= pattern_stats[label]['patterns'][pattern]['frequency']
        
        # 特征差异显著性
        feature_diff_score = pattern_significance[pattern_class][pattern]['feature_diff']
        
        # 预测准确性 - 包含该模式的序列被预测为其类别的概率
        prediction_score = pattern_prediction_stats[pattern]['pred_counts'][pattern_class] / 100
        
        # 综合分数 - 简单加权平均
        overall_score = (0.3 * discrimination_score + 
                        0.3 * feature_diff_score + 
                        0.4 * prediction_score)
        
        pattern_overall_score[pattern] = {
            'discrimination': discrimination_score,
            'feature_diff': feature_diff_score,
            'prediction': prediction_score,
            'overall': overall_score,
            'pattern_class': pattern_class,
            'is_known_pattern': pattern in class_patterns.get(pattern_class, [])
        }
    
    # 将综合评估结果写入文件
    with open(os.path.join(save_dir, 'pattern_overall_score.txt'), 'w') as f:
        f.write("模式综合评估结果\n")
        f.write("==============\n\n")
        
        # 按总分排序
        sorted_patterns = sorted(
            pattern_overall_score.items(),
            key=lambda x: x[1]['overall'],
            reverse=True
        )
        
        f.write("所有模式按综合得分排序:\n")
        f.write(f"{'模式':<10} {'类别':<8} {'已知':<6} {'区分能力':<10} {'特征差异':<10} {'预测准确性':<10} {'综合得分':<10}\n")
        f.write("-" * 70 + "\n")
        
        for pattern, scores in sorted_patterns:
            is_known = "是" if scores['is_known_pattern'] else "否"
            f.write(f"{pattern:<10} {scores['pattern_class']:<8} {is_known:<6} "
                   f"{scores['discrimination']:.4f}  {scores['feature_diff']:.4f}  "
                   f"{scores['prediction']:.4f}  {scores['overall']:.4f}\n")
        
        f.write("\n\n")
        
        # 按类别分组显示
        for label in unique_classes:
            f.write(f"类别{label}的模式评分:\n")
            f.write(f"{'模式':<10} {'已知':<6} {'区分能力':<10} {'特征差异':<10} {'预测准确性':<10} {'综合得分':<10}\n")
            f.write("-" * 65 + "\n")
            
            # 筛选该类别的模式并排序
            class_patterns_scores = [
                (p, s) for p, s in pattern_overall_score.items() 
                if s['pattern_class'] == label
            ]
            class_patterns_scores.sort(key=lambda x: x[1]['overall'], reverse=True)
            
            for pattern, scores in class_patterns_scores:
                is_known = "是" if scores['is_known_pattern'] else "否"
                f.write(f"{pattern:<10} {is_known:<6} "
                       f"{scores['discrimination']:.4f}  {scores['feature_diff']:.4f}  "
                       f"{scores['prediction']:.4f}  {scores['overall']:.4f}\n")
            
            f.write("\n")
    
    # 绘制综合评分图表
    plt.figure(figsize=(15, 10))
    
    # 按总分排序
    sorted_patterns = sorted(
        pattern_overall_score.items(),
        key=lambda x: x[1]['overall'],
        reverse=True
    )
    patterns = [p[0] for p in sorted_patterns]
    
    discrimination_scores = [pattern_overall_score[p]['discrimination'] for p in patterns]
    feature_diff_scores = [pattern_overall_score[p]['feature_diff'] for p in patterns]
    prediction_scores = [pattern_overall_score[p]['prediction'] for p in patterns]
    overall_scores = [pattern_overall_score[p]['overall'] for p in patterns]
    is_known = [pattern_overall_score[p]['is_known_pattern'] for p in patterns]
    
    x = np.arange(len(patterns))
    width = 0.2
    
    # 创建根据是否已知模式的不同颜色条
    bar_colors = ['green' if known else 'lightgray' for known in is_known]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 上半部分：各分项得分
    ax1.bar(x - width, discrimination_scores, width, label='区分能力', color='skyblue')
    ax1.bar(x, feature_diff_scores, width, label='特征差异', color='orange')
    ax1.bar(x + width, prediction_scores, width, label='预测准确性', color='purple')
    
    # 标记已知模式
    for i, known in enumerate(is_known):
        if known:
            ax1.axvspan(i - 0.4, i + 0.4, alpha=0.2, color='green')
    
    ax1.set_ylabel('分项得分')
    ax1.set_title('模式评估分项得分')
    ax1.set_xticks(x)
    ax1.set_xticklabels(patterns, rotation=45, ha='right')
    ax1.legend()
    
    # 下半部分：综合得分
    bars = ax2.bar(x, overall_scores, 0.6, label='综合得分', color=bar_colors)
    
    # 添加图例以区分已知和发现的模式
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='已知模式'),
        Patch(facecolor='lightgray', label='新发现模式')
    ]
    ax2.legend(handles=legend_elements)
    
    ax2.set_ylabel('综合得分')
    ax2.set_title('模式综合评估得分')
    ax2.set_xticks(x)
    ax2.set_xticklabels(patterns, rotation=45, ha='right')
    
    # 添加分类标签
    for i, p in enumerate(patterns):
        pattern_class = pattern_overall_score[p]['pattern_class']
        ax2.text(i, overall_scores[i]/2, f'C{pattern_class}', 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pattern_overall_score.png'), dpi=200)
    plt.close()
    
    print(f"模式发现与验证分析结果已保存到 {save_dir} 目录")
    
    return {
        'pattern_stats': pattern_stats,
        'pattern_discrimination': pattern_discrimination,
        'pattern_significance': pattern_significance,
        'pattern_prediction_stats': pattern_prediction_stats,
        'pattern_overall_score': pattern_overall_score
    }


def visualize_feature_importance(
    test_libraries, test_labels, 
    embedding_model, extractor_model, class_patterns,
    device='cuda', samples_per_class=5, save_dir='results/feature_importance'
):
    """
    可视化TCR序列中哪些位点对分类决策最重要，并与已知的类别特征模式对比
    
    参数:
    - test_libraries: 测试集库
    - test_labels: 测试集标签
    - embedding_model: 氨基酸序列嵌入模型
    - extractor_model: 特征提取模型
    - class_patterns: 每个类别已知的特征模式
    - device: 使用的设备
    - samples_per_class: 每个类别选择的样本数量
    - save_dir: 保存可视化结果的目录
    """
    print("\n分析序列特征重要性...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将模型设置为评估模式
    embedding_model.eval()
    extractor_model.eval()
    
    # 识别数据集中的类别
    unique_classes = sorted(set(test_labels))
    
    # 按类别组织库
    class_libraries = {label: [] for label in unique_classes}
    for lib, label in zip(test_libraries, test_labels):
        class_libraries[label].append(lib)
    
    # 为每个类别选择样本
    selected_sequences = {label: [] for label in unique_classes}
    for label in unique_classes:
        # 从该类别中随机选择一些库
        if len(class_libraries[label]) > samples_per_class:
            selected_libs = random.sample(class_libraries[label], samples_per_class)
        else:
            selected_libs = class_libraries[label]
        
        # 从每个选定的库中随机选择一个TCR序列
        for lib in selected_libs:
            if lib:  # 确保库不是空的
                selected_sequences[label].append(random.choice(lib))
    
    # 创建多面板图形
    n_classes = len(unique_classes)
    fig, axes = plt.subplots(n_classes, samples_per_class, 
                            figsize=(4*samples_per_class, 3*n_classes))
    if n_classes == 1 and samples_per_class == 1:
        axes = np.array([[axes]])
    elif n_classes == 1:
        axes = np.array([axes])
    elif samples_per_class == 1:
        axes = axes.reshape(-1, 1)
    
    # 分析每个选定序列的重要性
    for c_idx, label in enumerate(unique_classes):
        for s_idx, sequence in enumerate(selected_sequences[label][:samples_per_class]):
            # 跳过空序列
            if not sequence:
                continue
                
            # 编码序列
            encoded_seq = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in sequence[:MAX_LENGTH]]
            padding_length = MAX_LENGTH - len(encoded_seq)
            if padding_length > 0:
                encoded_seq = encoded_seq + [AA_TO_IDX['-']] * padding_length
            
            # 转换为张量
            seq_tensor = torch.tensor([encoded_seq], dtype=torch.long).to(device)
            
            # 计算梯度 - 这里我们需要开启梯度计算
            seq_tensor.requires_grad = True
            
            # 前向传播
            embeddings = embedding_model(seq_tensor)
            features = extractor_model(embeddings)
            
            # 计算输出对输入的梯度
            # 由于我们只关心序列特征，不需要实际的聚合和分类步骤
            # 我们可以使用特征的L2范数来衡量每个位置的重要性
            feature_norm = torch.norm(features, p=2)
            feature_norm.backward()
            
            # 提取梯度
            grads = seq_tensor.grad.abs().squeeze(0).cpu().numpy()
            
            # 将梯度归一化到[0,1]区间
            if grads.max() > grads.min():
                normalized_grads = (grads - grads.min()) / (grads.max() - grads.min())
            else:
                normalized_grads = np.zeros_like(grads)
            
            # 获取序列(不含填充)
            seq_len = len(sequence)
            seq_chars = list(sequence)
            importance = normalized_grads[:seq_len]
            
            # 绘制位点重要性
            ax = axes[c_idx, s_idx]
            bars = ax.bar(range(seq_len), importance, color='skyblue')
            
            # 添加氨基酸标签
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(seq_chars, rotation=45, fontsize=10)
            
            # 标记类别特征模式
            for pattern in class_patterns[label]:
                pattern_pos = sequence.find(pattern)
                if pattern_pos != -1:
                    ax.axvspan(pattern_pos, pattern_pos + len(pattern) - 0.5, 
                              alpha=0.3, color=f'C{label}',
                              label=f"类别{label}模式")
            
            # 添加标题和轴标签
            ax.set_title(f'类别{label} - 样本{s_idx+1}')
            ax.set_xlabel('序列位置')
            ax.set_ylabel('重要性分数')
            
            # 检查特征模式是否被正确识别为重要位点
            pattern_importance = []
            for pattern in class_patterns[label]:
                pattern_pos = sequence.find(pattern)
                if pattern_pos != -1:
                    pattern_end = pattern_pos + len(pattern)
                    pattern_imp = importance[pattern_pos:pattern_end].mean()
                    pattern_importance.append((pattern, pattern_imp))
            
            # 在图中标注模式的重要性分数
            if pattern_importance:
                for i, (pattern, imp) in enumerate(pattern_importance):
                    ax.text(0.05, 0.95-i*0.1, f"{pattern}: {imp:.3f}", 
                           transform=ax.transAxes, va='top')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sequence_importance.png'), dpi=200)
    plt.close()
    
    # 生成每个类别的重要性统计报告
    with open(os.path.join(save_dir, 'importance_report.txt'), 'w') as f:
        f.write("序列特征重要性分析报告\n")
        f.write("=========================\n\n")
        
        for label in unique_classes:
            f.write(f"类别 {label} 特征模式重要性:\n")
            f.write("--------------------------\n")
            
            all_pattern_imps = {}
            for sequence in selected_sequences[label]:
                if not sequence:
                    continue
                    
                # 编码序列
                encoded_seq = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in sequence[:MAX_LENGTH]]
                padding_length = MAX_LENGTH - len(encoded_seq)
                if padding_length > 0:
                    encoded_seq = encoded_seq + [AA_TO_IDX['-']] * padding_length
                
                seq_tensor = torch.tensor([encoded_seq], dtype=torch.long).to(device)
                seq_tensor.requires_grad = True
                
                # 前向和反向传播计算梯度
                embeddings = embedding_model(seq_tensor)
                features = extractor_model(embeddings)
                feature_norm = torch.norm(features, p=2)
                feature_norm.backward()
                
                grads = seq_tensor.grad.abs().squeeze(0).cpu().numpy()
                seq_len = len(sequence)
                if grads[:seq_len].max() > grads[:seq_len].min():
                    importance = (grads[:seq_len] - grads[:seq_len].min()) / (grads[:seq_len].max() - grads[:seq_len].min())
                else:
                    importance = np.zeros(seq_len)
                
                # 计算每个特征模式的平均重要性
                for pattern in class_patterns[label]:
                    pattern_pos = sequence.find(pattern)
                    if pattern_pos != -1:
                        pattern_end = pattern_pos + len(pattern)
                        pattern_imp = importance[pattern_pos:pattern_end].mean()
                        
                        if pattern not in all_pattern_imps:
                            all_pattern_imps[pattern] = []
                        all_pattern_imps[pattern].append(pattern_imp)
            
            # 输出每个模式的平均重要性分数
            for pattern, imps in all_pattern_imps.items():
                avg_imp = sum(imps) / len(imps)
                f.write(f"模式 '{pattern}' 的平均重要性分数: {avg_imp:.4f}\n")
            f.write("\n")
    
    # 创建综合可视化：比较不同类别模式的平均重要性
    pattern_importance_summary = {}
    for label in unique_classes:
        pattern_importance_summary[label] = {}
        for sequence in selected_sequences[label]:
            if not sequence:
                continue
                
            # 编码并计算重要性
            encoded_seq = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in sequence[:MAX_LENGTH]]
            padding_length = MAX_LENGTH - len(encoded_seq)
            if padding_length > 0:
                encoded_seq = encoded_seq + [AA_TO_IDX['-']] * padding_length
                
            seq_tensor = torch.tensor([encoded_seq], dtype=torch.long).to(device)
            seq_tensor.requires_grad = True
            
            embeddings = embedding_model(seq_tensor)
            features = extractor_model(embeddings)
            feature_norm = torch.norm(features, p=2)
            feature_norm.backward()
            
            grads = seq_tensor.grad.abs().squeeze(0).cpu().numpy()
            seq_len = len(sequence)
            if grads[:seq_len].max() > grads[:seq_len].min():
                importance = (grads[:seq_len] - grads[:seq_len].min()) / (grads[:seq_len].max() - grads[:seq_len].min())
            else:
                importance = np.zeros(seq_len)
            
            # 计算每个类别的每个模式的重要性
            for c_label in unique_classes:
                for pattern in class_patterns[c_label]:
                    pattern_pos = sequence.find(pattern)
                    if pattern_pos != -1:
                        pattern_end = pattern_pos + len(pattern)
                        pattern_imp = importance[pattern_pos:pattern_end].mean()
                        
                        key = f"类别{c_label}_{pattern}"
                        if key not in pattern_importance_summary[label]:
                            pattern_importance_summary[label][key] = []
                        pattern_importance_summary[label][key].append(pattern_imp)
    
    # 绘制模式重要性汇总图
    plt.figure(figsize=(12, 8))
    
    all_patterns = []
    for c_label in unique_classes:
        for pattern in class_patterns[c_label]:
            all_patterns.append(f"类别{c_label}_{pattern}")
    
    x = np.arange(len(all_patterns))
    width = 0.8 / len(unique_classes)
    
    for i, label in enumerate(unique_classes):
        means = []
        stds = []
        for pattern in all_patterns:
            if pattern in pattern_importance_summary[label]:
                values = pattern_importance_summary[label][pattern]
                means.append(np.mean(values) if values else 0)
                stds.append(np.std(values) if values and len(values) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
        
        plt.bar(x + i*width - width*(len(unique_classes)-1)/2, means, width, 
               label=f'类别{label}序列', yerr=stds, capsize=5)
    
    plt.xlabel('模式')
    plt.ylabel('平均重要性分数')
    plt.title('不同类别序列对各模式的敏感性')
    plt.xticks(x, all_patterns, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pattern_importance_summary.png'), dpi=200)
    plt.close()
    
    print(f"特征重要性分析结果已保存到 {save_dir} 目录")
    
    return pattern_importance_summary




def evaluate_model(
    test_libraries, test_labels,
    embedding_model, extractor_model, aggregator, classifier_model,
    device='cuda', batch_size=128, visualize=True
):
    """
    在测试集上评估完整模型的性能
    
    参数:
    - test_libraries: 测试集库
    - test_labels: 测试集标签
    - embedding_model: 氨基酸序列嵌入模型
    - extractor_model: 特征提取模型
    - aggregator: 库特征聚合器
    - classifier_model: 分类器模型
    - device: 评估设备
    - batch_size: 批量大小
    - visualize: 是否生成可视化结果
    
    返回:
    - test_accuracy: 测试准确率
    - test_f1: F1分数
    - confusion_matrix: 混淆矩阵
    """
    print("\n开始模型评估...")
    
    # 将所有模型设置为评估模式
    embedding_model.eval()
    extractor_model.eval()
    aggregator.eval()
    classifier_model.eval()
    
    # 将模型移至指定设备
    embedding_model = embedding_model.to(device)
    extractor_model = extractor_model.to(device)
    aggregator = aggregator.to(device)
    classifier_model = classifier_model.to(device)
    
    # 用于收集结果
    all_predictions = []
    all_true_labels = []
    correct_count = 0
    
    # 评估进度条
    progress_bar = tqdm(enumerate(zip(test_libraries, test_labels)), 
                       desc="评估", total=len(test_libraries))
    
    with torch.no_grad():  # 禁用梯度计算
        for lib_idx, (library, true_label) in progress_bar:
            # 处理库中的序列
            library_sequences = []
            for seq in library:
                # 编码序列
                encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                padding_length = MAX_LENGTH - len(encoded)
                if padding_length > 0:
                    encoded = encoded + [AA_TO_IDX['-']] * padding_length
                library_sequences.append(encoded)
            
            if len(library_sequences) == 0:
                continue  # 跳过空库
                
            # 分批处理序列以避免内存问题
            batch_features = []
            
            for i in range(0, len(library_sequences), batch_size):
                batch_seqs = library_sequences[i:min(i+batch_size, len(library_sequences))]
                batch_tensor = torch.tensor(batch_seqs, dtype=torch.long).to(device)
                
                # 序列嵌入和特征提取
                embeddings = embedding_model(batch_tensor)
                features = extractor_model(embeddings)
                batch_features.append(features)
            
            # 合并所有批次的特征
            all_features = torch.cat(batch_features, dim=0)
            
            # 使用聚合器生成库特征
            mask = torch.ones(1, all_features.shape[0], device=device)
            library_feature = aggregator(all_features, mask)
            
            # 分类预测
            logits = classifier_model(library_feature)
            _, predicted = torch.max(logits, 1)
            predicted_label = predicted.item()
            
            # 记录预测和真实标签
            all_predictions.append(predicted_label)
            all_true_labels.append(true_label)
            
            # 统计正确预测
            if predicted_label == true_label:
                correct_count += 1
                
            # 更新进度条
            progress_bar.set_postfix({
                'acc': f"{100.0 * correct_count / (lib_idx + 1):.2f}%"
            })
    
    # 计算评估指标
    test_accuracy = 100.0 * correct_count / len(test_libraries)
    test_f1 = f1_score(all_true_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    
    # 输出评估结果
    print("\n测试评估结果:")
    print(f"准确率: {test_accuracy:.2f}%")
    print(f"F1分数: {test_f1:.4f}")
    print("混淆矩阵:")
    print(conf_matrix)
    
    # 可视化混淆矩阵
    if visualize:
        plt.figure(figsize=(10, 8))
        class_names = [f'类别{i}' for i in range(len(np.unique(all_true_labels)))]
        
        # 绘制混淆矩阵
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('测试集混淆矩阵')
        plt.tight_layout()
        plt.savefig('results/test_confusion_matrix.png')
        
        # 生成分类报告并保存
        from sklearn.metrics import classification_report
        report = classification_report(all_true_labels, all_predictions, 
                                      target_names=class_names, digits=3)
        print("\n分类报告:")
        print(report)
        
        with open('results/classification_report.txt', 'w') as f:
            f.write(report)
        
        # 生成ROC曲线 (仅适用于二分类情况)
        if len(np.unique(all_true_labels)) == 2:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            # 获取样本的概率输出
            all_probs = []
            with torch.no_grad():
                for lib_idx, (library, _) in enumerate(zip(test_libraries, test_labels)):
                    library_sequences = []
                    for seq in library:
                        encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                        padding_length = MAX_LENGTH - len(encoded)
                        if padding_length > 0:
                            encoded = encoded + [AA_TO_IDX['-']] * padding_length
                        library_sequences.append(encoded)
                    
                    batch_features = []
                    for i in range(0, len(library_sequences), batch_size):
                        batch_seqs = library_sequences[i:min(i+batch_size, len(library_sequences))]
                        batch_tensor = torch.tensor(batch_seqs, dtype=torch.long).to(device)
                        embeddings = embedding_model(batch_tensor)
                        features = extractor_model(embeddings)
                        batch_features.append(features)
                    
                    all_features = torch.cat(batch_features, dim=0)
                    mask = torch.ones(1, all_features.shape[0], device=device)
                    library_feature = aggregator(all_features, mask)
                    logits = classifier_model(library_feature)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0, 1]  # 获取类别1的概率
                    all_probs.append(probs)
                    
            # 计算ROC曲线和AUC
            fpr, tpr, _ = roc_curve(all_true_labels, all_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC 曲线 (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率')
            plt.ylabel('真阳性率')
            plt.title('受试者工作特征曲线')
            plt.legend(loc="lower right")
            plt.savefig('results/roc_curve.png')
            
    # 返回评估指标
    return test_accuracy, test_f1, conf_matrix




def train_classifier_with_aggregator(
    train_libraries, val_libraries, test_libraries,
    train_labels, val_labels, test_labels,
    embedding_model, extractor_model, aggregator, classifier_model,
    finetune_extractor=False,
    device='cuda', batch_size=32,
    learning_rate=0.0005, epochs=50
):
    """
    在预训练的特征提取器基础上训练库聚合器和分类器
    
    参数:
    - train/val/test_libraries: 训练/验证/测试集库
    - train/val/test_labels: 训练/验证/测试集标签
    - embedding_model: 预训练的氨基酸序列嵌入模型
    - extractor_model: 预训练的特征提取器
    - aggregator: 库特征聚合器
    - classifier_model: 分类器模型
    - finetune_extractor: 是否微调特征提取器
    - device: 训练设备
    - batch_size: 批大小
    - learning_rate: 学习率
    - epochs: 训练轮数
    
    返回:
    - 训练后的模型和评估结果
    """
    print(f"\n开始训练分类器和聚合器{' (同时微调特征提取器)' if finetune_extractor else ''}...")
    
    # 将模型移至指定设备
    embedding_model = embedding_model.to(device)
    extractor_model = extractor_model.to(device)
    aggregator = aggregator.to(device)
    classifier_model = classifier_model.to(device)
    
    # 设置特征提取器是否可训练
    for param in embedding_model.parameters():
        param.requires_grad = finetune_extractor
    for param in extractor_model.parameters():
        param.requires_grad = finetune_extractor
    
    # 要优化的参数
    if finetune_extractor:
        parameters = list(embedding_model.parameters()) + \
                    list(extractor_model.parameters()) + \
                    list(aggregator.parameters()) + \
                    list(classifier_model.parameters())
    else:
        parameters = list(aggregator.parameters()) + list(classifier_model.parameters())
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(parameters, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 用于早停的变量
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 10
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        # 训练阶段
        if finetune_extractor:
            embedding_model.train()
            extractor_model.train()
        else:
            embedding_model.eval()
            extractor_model.eval()
        
        aggregator.train()
        classifier_model.train()
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # 遍历训练集的每个库
        for lib_idx, (library, label) in enumerate(tqdm(zip(train_libraries, train_labels), 
                                                      desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            # 处理库中的序列
            library_sequences = []
            for seq in library:
                encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                padding_length = MAX_LENGTH - len(encoded)
                if padding_length > 0:
                    encoded = encoded + [AA_TO_IDX['-']] * padding_length
                library_sequences.append(encoded)
            
            # 分批处理以避免内存问题
            batch_features = []
            for i in range(0, len(library_sequences), batch_size):
                batch_seqs = library_sequences[i:min(i+batch_size, len(library_sequences))]
                batch_seqs = torch.tensor(batch_seqs, dtype=torch.long).to(device)
                
                # 序列嵌入和特征提取
                with torch.set_grad_enabled(finetune_extractor):  # 只有在微调时才计算梯度
                    embeddings = embedding_model(batch_seqs)
                    features = extractor_model(embeddings)
                    batch_features.append(features)
            
            # 合并批次特征
            all_features = torch.cat(batch_features, dim=0)
            
            # 使用聚合器聚合特征
            mask = torch.ones(1, all_features.shape[0], device=device)
            lib_feature = aggregator(all_features, mask)
            
            # 分类
            output = classifier_model(lib_feature)
            target = torch.tensor([label], dtype=torch.long).to(device)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            epoch_total += 1
            epoch_correct += (predicted == target).sum().item()
        
        # 计算训练指标
        train_loss = epoch_loss / len(train_libraries)
        train_acc = epoch_correct / epoch_total * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        embedding_model.eval()
        extractor_model.eval()
        aggregator.eval()
        classifier_model.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for lib_idx, (library, label) in enumerate(tqdm(zip(val_libraries, val_labels), 
                                                         desc=f'Epoch {epoch+1}/{epochs} [Val]')):
                # 处理验证集中的单个库
                library_sequences = []
                for seq in library:
                    encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                    padding_length = MAX_LENGTH - len(encoded)
                    if padding_length > 0:
                        encoded = encoded + [AA_TO_IDX['-']] * padding_length
                    library_sequences.append(encoded)
                
                # 分批处理
                batch_features = []
                for i in range(0, len(library_sequences), batch_size):
                    batch_seqs = library_sequences[i:min(i+batch_size, len(library_sequences))]
                    batch_seqs = torch.tensor(batch_seqs, dtype=torch.long).to(device)
                    
                    embeddings = embedding_model(batch_seqs)
                    features = extractor_model(embeddings)
                    batch_features.append(features)
                
                # 合并批次特征
                all_features = torch.cat(batch_features, dim=0)
                
                # 聚合特征
                mask = torch.ones(1, all_features.shape[0], device=device)
                lib_feature = aggregator(all_features, mask)
                
                # 分类
                output = classifier_model(lib_feature)
                target = torch.tensor([label], dtype=torch.long).to(device)
                
                # 计算损失
                loss = criterion(output, target)
                val_loss += loss.item()
                
                # 统计
                _, predicted = torch.max(output.data, 1)
                val_total += 1
                val_correct += (predicted == target).sum().item()
        
        # 计算验证指标
        val_loss = val_loss / len(val_libraries)
        val_acc = val_correct / val_total * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"Epoch {epoch+1}/{epochs} - 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # 保存最佳模型
            model_filename = f"models/best_classifier_model_finetune_{finetune_extractor}.pt"
            torch.save({
                'embedding_model': embedding_model.state_dict(),
                'extractor_model': extractor_model.state_dict(),
                'aggregator': aggregator.state_dict(),
                'classifier_model': classifier_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, model_filename)
            
            print(f"模型已保存: {model_filename}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"早停: {early_stopping_patience}个epoch没有改善")
                break
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/classifier_training_finetune_{finetune_extractor}.png')
    plt.close()
    
    # 加载最佳模型
    model_filename = f"models/best_classifier_model_finetune_{finetune_extractor}.pt"
    checkpoint = torch.load(model_filename)
    embedding_model.load_state_dict(checkpoint['embedding_model'])
    extractor_model.load_state_dict(checkpoint['extractor_model'])
    aggregator.load_state_dict(checkpoint['aggregator'])
    classifier_model.load_state_dict(checkpoint['classifier_model'])
    
    print(f"已加载最佳模型，验证准确率: {checkpoint['val_acc']:.2f}%")
    
    # 测试阶段
    embedding_model.eval()
    extractor_model.eval()
    aggregator.eval()
    classifier_model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for lib_idx, (library, label) in enumerate(tqdm(zip(test_libraries, test_labels), 
                                                     desc='测试中...')):
            # 处理库中的序列
            library_sequences = []
            for seq in library:
                encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                padding_length = MAX_LENGTH - len(encoded)
                if padding_length > 0:
                    encoded = encoded + [AA_TO_IDX['-']] * padding_length
                library_sequences.append(encoded)
            
            # 分批处理
            batch_features = []
            for i in range(0, len(library_sequences), batch_size):
                batch_seqs = library_sequences[i:min(i+batch_size, len(library_sequences))]
                batch_seqs = torch.tensor(batch_seqs, dtype=torch.long).to(device)
                
                embeddings = embedding_model(batch_seqs)
                features = extractor_model(embeddings)
                batch_features.append(features)
            
            # 合并批次特征
            all_features = torch.cat(batch_features, dim=0)
            
            # 聚合特征
            mask = torch.ones(1, all_features.shape[0], device=device)
            lib_feature = aggregator(all_features, mask)
            
            # 分类
            output = classifier_model(lib_feature)
            target = torch.tensor([label], dtype=torch.long).to(device)
            
            # 统计
            _, predicted = torch.max(output.data, 1)
            test_total += 1
            test_correct += (predicted == target).sum().item()
            
            all_preds.append(predicted.cpu().item())
            all_labels.append(label)
    
    # 计算测试指标
    test_acc = test_correct / test_total * 100
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"\n测试结果 (微调={finetune_extractor}):")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"测试F1分数: {test_f1:.4f}")
    print("混淆矩阵:")
    print(conf_matrix)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'类别{i}' for i in range(len(np.unique(all_labels)))],
                yticklabels=[f'类别{i}' for i in range(len(np.unique(all_labels)))])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'混淆矩阵 (微调={finetune_extractor})')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_finetune_{finetune_extractor}.png')
    plt.close()
    
    return (embedding_model, extractor_model, aggregator, classifier_model, 
            {'test_acc': test_acc, 'test_f1': test_f1, 'conf_matrix': conf_matrix,
             'train_losses': train_losses, 'val_losses': val_losses, 
             'train_accs': train_accs, 'val_accs': val_accs})



def pretrain_contrastive_learning(
    train_dataset, val_dataset,
    embedding_model, extractor_model,
    device='cuda', batch_size=128,
    learning_rate=0.001, epochs=30
):
    """
    使用对比学习预训练序列嵌入模型和特征提取器
    
    参数:
    - train_dataset: 训练数据集
    - val_dataset: 验证数据集
    - embedding_model: 氨基酸序列嵌入模型
    - extractor_model: 特征提取模型
    - device: 训练设备
    - batch_size: 批大小
    - learning_rate: 学习率
    - epochs: 训练轮数
    
    返回:
    - 预训练后的嵌入模型和特征提取器
    """
    print("\n开始对比学习预训练...")
    
    # 将模型移至指定设备
    embedding_model = embedding_model.to(device)
    extractor_model = extractor_model.to(device)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=val_dataset.collate_fn)
    

    
    # 定义损失函数和优化器
    criterion = NTXentLoss(temperature=0.07)
    optimizer = optim.Adam(
        list(embedding_model.parameters()) + list(extractor_model.parameters()), 
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 用于早停的变量
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        embedding_model.train()
        extractor_model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for sequences, lib_ids, _ in progress_bar:
            # 移动数据到设备
            sequences = sequences.to(device)
            lib_ids = lib_ids.to(device)
            
            # 前向传播
            embeddings = embedding_model(sequences)
            features = extractor_model(embeddings)
            
            # 计算对比损失
            loss = criterion(features, lib_ids)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        embedding_model.eval()
        extractor_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for sequences, lib_ids, _ in progress_bar:
                sequences = sequences.to(device)
                lib_ids = lib_ids.to(device)
                
                embeddings = embedding_model(sequences)
                features = extractor_model(embeddings)
                
                loss = criterion(features, lib_ids)
                val_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'embedding_model': embedding_model.state_dict(),
                'extractor_model': extractor_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, 'models/best_pretrained_model.pt')
            
            print(f'模型已保存: models/best_pretrained_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'早停: {early_stopping_patience} 个epoch没有改善')
                break
    
    # 绘制训练过程
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('对比学习损失')
    plt.title('预训练对比学习损失')
    plt.legend()
    plt.savefig('results/pretraining_loss.png')
    plt.close()
    
    print("预训练完成!")
    
    # 加载最佳模型
    checkpoint = torch.load('models/best_pretrained_model.pt')
    embedding_model.load_state_dict(checkpoint['embedding_model'])
    extractor_model.load_state_dict(checkpoint['extractor_model'])
    
    return embedding_model, extractor_model, train_losses, val_losses






def single_stage_training(
    train_dataset, val_dataset, test_dataset,
    train_libraries, val_libraries, test_libraries,
    train_labels, val_labels, test_labels,
    embedding_model, extractor_model, aggregator, classifier_model,
    device='cuda', batch_size=128, 
    learning_rate=0.001, epochs=80
):
    """
    单阶段端到端训练整个模型 F(X) = Cls(Agg(Ext(Emb(X))))
    """
    # 将模型移到设备
    embedding_model = embedding_model.to(device)
    extractor_model = extractor_model.to(device)
    aggregator = aggregator.to(device)
    classifier_model = classifier_model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        list(embedding_model.parameters()) +
        list(extractor_model.parameters()) +
        list(aggregator.parameters()) +
        list(classifier_model.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 用于早停的变量
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("\n开始单阶段训练...")
    
    for epoch in range(epochs):
        # 训练阶段 - 库级别训练
        embedding_model.train()
        extractor_model.train()
        aggregator.train()
        classifier_model.train()
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        # 对每个训练库进行处理
        for lib_idx, (library, label) in enumerate(tqdm(zip(train_libraries, train_labels), 
                                                      desc=f'Epoch {epoch+1}/{epochs} [Train]')):
            # 处理单个库
            library_sequences = []
            for seq in library:
                # 编码序列
                encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                padding_length = MAX_LENGTH - len(encoded)
                if padding_length > 0:
                    encoded = encoded + [AA_TO_IDX['-']] * padding_length
                library_sequences.append(encoded)
            
            # 分批处理库中的序列以避免内存问题
            batch_features = []
            for i in range(0, len(library_sequences), batch_size):
                batch_seqs = library_sequences[i:min(i+batch_size, len(library_sequences))]
                batch_seqs = torch.tensor(batch_seqs, dtype=torch.long).to(device)
                
                # 序列嵌入和特征提取
                with torch.set_grad_enabled(True):
                    embeddings = embedding_model(batch_seqs)
                    features = extractor_model(embeddings)
                    batch_features.append(features)
            
            # 合并所有批次的特征
            all_features = torch.cat(batch_features, dim=0)
            
            # 使用聚合器聚合库内所有序列特征
            mask = torch.ones(1, all_features.shape[0], device=device)
            lib_feature = aggregator(all_features, mask)
            
            # 分类
            output = classifier_model(lib_feature)
            target = torch.tensor([label], dtype=torch.long).to(device)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            epoch_total += 1
            epoch_correct += (predicted == target).sum().item()
        
        # 计算训练指标
        train_loss = epoch_loss / len(train_libraries)
        train_acc = epoch_correct / epoch_total * 100
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        embedding_model.eval()
        extractor_model.eval()
        aggregator.eval()
        classifier_model.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for lib_idx, (library, label) in enumerate(tqdm(zip(val_libraries, val_labels), 
                                                         desc=f'Epoch {epoch+1}/{epochs} [Val]')):
                # 处理验证集中的单个库
                library_sequences = []
                for seq in library:
                    encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
                    padding_length = MAX_LENGTH - len(encoded)
                    if padding_length > 0:
                        encoded = encoded + [AA_TO_IDX['-']] * padding_length
                    library_sequences.append(encoded)
                
                # 分批处理
                batch_features = []
                for i in range(0, len(library_sequences), batch_size):
                    batch_seqs = library_sequences[i:min(i+batch_size, len(library_sequences))]
                    batch_seqs = torch.tensor(batch_seqs, dtype=torch.long).to(device)
                    
                    embeddings = embedding_model(batch_seqs)
                    features = extractor_model(embeddings)
                    batch_features.append(features)
                
                # 合并所有批次的特征
                all_features = torch.cat(batch_features, dim=0) if batch_features else torch.tensor([], device=device)
                
                if len(all_features) > 0:
                    # 聚合库特征
                    mask = torch.ones(1, all_features.shape[0], device=device)
                    lib_feature = aggregator(all_features, mask)
                    
                    # 分类
                    output = classifier_model(lib_feature)
                    target = torch.tensor([label], dtype=torch.long).to(device)
                    
                    # 计算损失
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    # 统计
                    _, predicted = torch.max(output.data, 1)
                    val_total += 1
                    val_correct += (predicted == target).sum().item()
        
        # 计算验证指标
        val_loss = val_loss / len(val_libraries)
        val_acc = val_correct / val_total * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"Epoch {epoch+1}/{epochs} - 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'embedding_model': embedding_model.state_dict(),
                'extractor_model': extractor_model.state_dict(),
                'aggregator': aggregator.state_dict(),
                'classifier_model': classifier_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, 'models/best_single_stage_model.pt')
            
            print(f"模型已保存: models/best_single_stage_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"早停: {early_stopping_patience}个epoch没有改善")
                break
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/single_stage_training.png')
    plt.close()
    
    # 加载最佳模型
    checkpoint = torch.load('models/best_single_stage_model.pt')
    embedding_model.load_state_dict(checkpoint['embedding_model'])
    extractor_model.load_state_dict(checkpoint['extractor_model'])
    aggregator.load_state_dict(checkpoint['aggregator'])
    classifier_model.load_state_dict(checkpoint['classifier_model'])
    
    print(f"已加载最佳模型，验证准确率: {checkpoint['val_acc']:.2f}%")
    
    # 返回训练好的模型和训练统计信息
    return (embedding_model, extractor_model, aggregator, classifier_model, 
            {'train_losses': train_losses, 'val_losses': val_losses, 
             'train_accs': train_accs, 'val_accs': val_accs})

def main(data_path='data/tcr_small_dataset.pkl', 
         training_mode='single_stage',    # 'single_stage' 或 'two_stage'
         finetune_extractor=False,        # 在two_stage模式下是否微调Ext
         device='cuda', 
         feature_dim=128,
         batch_size=128,
         learning_rate=0.001,
         pretrain_epochs=30,
         finetune_epochs=50,
         single_stage_epochs=80,
         aggregation_type='multi_head'):  # 'mean', 'attention', 'multi_head', 'adaptive'
    """
    TCR免疫库分类训练主函数
    
    参数:
    - data_path: 数据集路径
    - training_mode: 训练模式，'single_stage'为端到端训练，'two_stage'为两阶段训练
    - finetune_extractor: 在two_stage模式下，是否微调特征提取器
    - device: 训练设备 'cuda'或'cpu'
    - feature_dim: 特征维度
    - batch_size: 批量大小
    - learning_rate: 学习率
    - pretrain_epochs: 预训练阶段的训练轮数
    - finetune_epochs: 微调/分类阶段的训练轮数
    - single_stage_epochs: 单阶段训练的轮数
    - aggregation_type: 库内特征聚合方法
    """
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    # 加载数据
    print("加载数据...")
    with open(data_path, 'rb') as f:
        libraries, labels = pickle.load(f)
    
    print(f"数据集信息: {len(libraries)} 个库, 每个库有 {len(libraries[0])} 条TCR序列")
    print(f"类别分布: {np.bincount(labels)}")
    
    # 在库级别划分数据集 - 保持一致的数据划分
    print("\n划分数据集...")
    train_indices, temp_indices = train_test_split(
        range(len(libraries)), test_size=0.3, random_state=42, stratify=labels
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, stratify=[labels[i] for i in temp_indices]
    )
    
    # 创建库的训练/验证/测试集
    train_libraries = [libraries[i] for i in train_indices]
    val_libraries = [libraries[i] for i in val_indices]
    test_libraries = [libraries[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"训练集: {len(train_libraries)} 个库")
    print(f"验证集: {len(val_libraries)} 个库")
    print(f"测试集: {len(test_libraries)} 个库")
    
    # 创建序列级别的数据集
    train_dataset = TCRDataset(train_libraries, train_labels)
    val_dataset = TCRDataset(val_libraries, val_labels)
    test_dataset = TCRDataset(test_libraries, test_labels)
    
    # 创建模型
    print("创建模型...")
    embedding_model = TCREmbedding(vocab_size=len(AA_LETTERS), embed_dim=32)
    extractor_model = TCRExtractor(in_channels=32, feature_dim=feature_dim)
    aggregator = TCRAggregator(feature_dim=feature_dim, aggregation_type=aggregation_type)
    classifier_model = TCRClassifier(input_dim=feature_dim, num_classes=len(np.unique(labels)))
    
    # 创建模型保存目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 获取数据集生成时使用的类别模式 (需要从数据生成函数中提取)
    class_patterns = []
    for c in range(len(np.unique(labels))):
        if c == 0:
            class_patterns.append(['AKII','EVDWQ','ADHI','VHVQ','RWKAG'])  # 类别0的特征模式
        elif c == 1:
            class_patterns.append(['MKFHMED','ENNYKCR','PDVL','WHDCIL','IEPK'])  # 类别1的特征模式
        elif c == 2:
            class_patterns.append(['GNNHKD','VIGRP','VIMCIC','PKDHWM'])
        elif c == 3:
            class_patterns.append(['PRFKFIV','WQWPNI','TSDCE'])
        
        
    # 根据选择的训练模式执行相应流程
    if training_mode == 'single_stage':
        print("\n执行单阶段训练流程...")
        # 单阶段训练 - 直接端到端训练整个模型
        
        single_stage_training(
            train_dataset, val_dataset, test_dataset,
            train_libraries, val_libraries, test_libraries,
            train_labels, val_labels, test_labels,
            embedding_model, extractor_model, aggregator, classifier_model,
            device=device, batch_size=batch_size, 
            learning_rate=learning_rate, epochs=single_stage_epochs
        )
        
        
    elif training_mode == 'two_stage':
        print("\n执行两阶段训练流程...")
        # 阶段1: 对比学习预训练特征提取器
        
        pretrain_contrastive_learning(
            train_dataset, val_dataset,
            embedding_model, extractor_model,
            device=device, batch_size=batch_size,
            learning_rate=learning_rate, epochs=pretrain_epochs
        )
        
        
        # 阶段2: 训练聚合器和分类器，可选择是否微调特征提取器
        
        train_classifier_with_aggregator(
            train_libraries, val_libraries, test_libraries,
            train_labels, val_labels, test_labels,
            embedding_model, extractor_model, aggregator, classifier_model,
            finetune_extractor=finetune_extractor,
            device=device, batch_size=batch_size,
            learning_rate=learning_rate, epochs=finetune_epochs
        )
        
    else:
        raise ValueError(f"不支持的训练模式: {training_mode}. 请选择 'single_stage' 或 'two_stage'")
    
    # 模型评估
    
    test_accuracy, test_f1, confusion = evaluate_model(
        test_libraries, test_labels,
        embedding_model, extractor_model, aggregator, classifier_model,
        device=device
    )
    
    
    # 模型解释性分析
    print("\n分析模型学习到的特征模式...")
    
    # 1. 特征重要性可视化
    visualize_feature_importance(
        test_libraries, test_labels, 
        embedding_model, extractor_model, class_patterns,
        device=device
    )
    

    
    # 3. 验证模型是否找到了类别模式
    pattern_discovery_analysis(
        test_libraries, test_labels,
        embedding_model, extractor_model,
        class_patterns, device=device
    )
    
    
    print("\n训练完成!")
    # print(f"最终测试准确率: {test_accuracy:.2f}%")
    # print(f"最终测试F1分数: {test_f1:.4f}")
    
    # 将结果保存到文件
    
    with open(f"results/results_{training_mode}_finetune_{finetune_extractor}.json", "w") as f:
        json.dump({
            "training_mode": training_mode,
            "finetune_extractor": finetune_extractor,
            "aggregation_type": aggregation_type,
            "test_accuracy": float(test_accuracy),
            "test_f1": float(test_f1),
            "confusion_matrix": confusion.tolist()
        }, f, indent=4)
    
    
    return embedding_model, extractor_model, aggregator, classifier_model

if __name__ == "__main__":
    # 运行单阶段训练
    main(training_mode='single_stage')
    
    # 运行两阶段训练，不微调特征提取器
    # main(training_mode='two_stage', finetune_extractor=False)
    
    # 运行两阶段训练，微调特征提取器
    # main(training_mode='two_stage', finetune_extractor=True)